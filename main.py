import os
import json
import yaml
from pathlib import Path
import pprint
import re

import click
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 내부 라이브러리 임포트
from src.agent.base import make_agent_pipeline
from src.prompt.base import PromptMaker
from src.utils import utils
from src.graph.sample_subgraph import ToolGraphSampler
from dotenv import load_dotenv
from src.prompt.domain_prompt import domain_prompt_dict


load_dotenv()

@click.command()
@click.option("--yaml_path", default=None, help="Path to a predefined YAML file.")
def main(yaml_path):

    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        try:
            agents_num = yaml_data["agent_num"]
            rounds_num = yaml_data["rounds_num"]
            fewshot = yaml_data["fewshot"]
            domain = yaml_data["domain"]
            output_path = yaml_data["output_path"]
            task = yaml_data["task"]
            dataset_num = yaml_data["dataset_num"]
        except KeyError as e:
            raise KeyError(f"Required key is missing in the YAML file: {e}")

    # create new output path every time
    unique_output_fp = utils.create_unique_output_path(output_path, task)

    with open("src/graph/tool_graph.json", "r", encoding="utf-8") as f:
        tool_graph = json.load(f)


    for i in range(dataset_num):
        function_list, function_dumps_per_dialogue = utils.sample_functions_from_graph_and_get_json(
            tool_graph, task, rounds_num
        )
        
        # Generate personas using function, domain_desc
        personas = utils.get_persona_prompts(agents_num, function_dumps_per_dialogue, domain_prompt_dict[domain])
        
        # 7. LangChain Prompt 설정
        pm = PromptMaker(
            agent_num=agents_num,
            rounds_num=rounds_num,
            fewshot=fewshot,
            function_dumps_per_dialogue=function_dumps_per_dialogue,
            domain=domain,
            task=task,
            personas=personas
        )

        for idx, persona in enumerate(personas):
            print(f"persona {idx} in main: {persona}")
            print('---'*10)
            
        print(f"\function_dumps_per_dialogue: {function_dumps_per_dialogue}\n")
        
        # 8. LangChain 파이프라인 생성
        main_graph = make_agent_pipeline(pm)

        # 9. 파이프라인 구조 시각화 후 저장
        utils.draw_langgraph(main_graph, Path(output_path))

        data_prompt = pm.data_prompt()
        messages = [HumanMessage(content=data_prompt)]

        rounds_events_list = []

        is_multi_round = True if task == "multi_round" else False
        actual_rounds = rounds_num if is_multi_round else 1

        next_parameter = None
        for rdx in range(actual_rounds):
            print(f"\n=== [Dataset {i+1}] Round {rdx+1} ===")
            round_function_list = function_list[rdx]

            round_function_json = utils.get_functions_from_tool_graph(
                round_function_list, json_file_path="src/graph/tool_graph.json"
            )

            # convert to function definition which is a str type
            round_function_def = json.dumps(
                round_function_json, ensure_ascii=False, indent=4
            )
            print(f"\nfunctions_per_round: {round_function_def}\n")

            # generate paramters
            if next_parameter:  # if it is not the first round
                round_parameters = utils.get_parameter_values(
                    functions=round_function_def,
                    target_function=next_parameter["target_function"],
                    target_parameter=next_parameter["target_parameter"],
                )
                print(
                    f"\nnext_parameter['target_function']: {next_parameter['target_function']}"
                )
                print(
                    f"next_parameter['target_parameter']: {next_parameter['target_parameter']}"
                )
            else:  # if it is the first round
                round_parameters = utils.get_parameter_values(round_function_def)

            print(f"\nround_parameters: {round_parameters}\n")

            # 함수 정보 추출
            parsed_details = utils.parse_json_functions(round_parameters)
            print(f"\nparsed_details: {parsed_details}\n")

            round_prompt = utils.system_prompt_per_round(
                round_function_def, round_parameters
            )
            messages.append(HumanMessage(content=round_prompt))

            events_gen = main_graph.stream(
                {"messages": messages},
                {"recursion_limit": 40},
            )

            # generator -> list로 변환
            rounds_events = list(events_gen)
            rounds_events_list.append(rounds_events)

            virtual_output = None
            if (
                is_multi_round and rdx < actual_rounds - 1
            ):  # if it is not the last round


                source_function_str = round_function_def
                print(f'source_function_str: {source_function_str}')
                source_parameter_value_str = round_parameters
                print(f'source_parameter_value_str: {source_parameter_value_str}')

                if task == "single_round":
                    break
                else:
                    virtual_output = utils.virtual_function_call(
                        source_function_str, source_parameter_value_str
                    )
                print(f"\nvirtual_output: {virtual_output}\n")

                if task == "multi_round":
                    next_parameter = {
                        "target_function": function_list[rdx + 1],
                        "target_parameter": virtual_output,
                    }

            if virtual_output:
                messages.append(
                    AIMessage(
                        content=f"[Return Value from Round {rdx+1}]: {virtual_output}\n"
                        "You should use the return value to progress the next round conversation so that the return value can be used as a parameter for the next function call.",
                        name="AI_Assistant",
                    )
                )

        # 11. 결과 JSON 저장
        save_dicts, metadata_dicts = utils.save_data(
            rounds_events_list, f"{unique_output_fp}.json"
        )

        # 12. 메타데이터 저장
        metadata_path = os.path.splitext(unique_output_fp)[0] + "_metadata.json"
        metadata = {
            "domain": domain,
            "round": rounds_num,
            "funclist": function_list,
            "parameter": "not yet decided",
            "orchestrator": metadata_dicts,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        print(f"Data saved to '{unique_output_fp}'")
        print(f"Metadata saved to '{metadata_path}'")


if __name__ == "__main__":
    main()