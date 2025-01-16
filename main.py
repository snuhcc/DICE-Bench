import os
import json
import yaml
from pathlib import Path
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

# ========= colorlog 세팅 ==========
import logging
from colorlog import ColoredFormatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = ColoredFormatter(
    "%(log_color)s[%(levelname)s]%(reset)s %(blue)s%(name)s:%(reset)s %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red'
    },
)
handler.setFormatter(formatter)
logger.addHandler(handler)
# ========================================

load_dotenv()

@click.command()
@click.option("--yaml_path", default=None, help="Path to a predefined YAML file.")
def main(yaml_path):
    agents_num, rounds_num, fewshot, domain, output_path, task, dataset_num = utils.load_yaml(yaml_path)

    # create new output path every time
    unique_output_fp = utils.create_unique_output_path(output_path, task)
    logger.info(f"Output file path: {unique_output_fp}")

    with open("src/graph/tool_graph.json", "r", encoding="utf-8") as f:
        tool_graph = json.load(f)

    # Initialize list to store all conversations
    all_conversations = []

    for i in range(dataset_num):
        logger.info(f"=== [Dataset {i+1}] Simulation Start ===")

        function_list, function_dumps_per_dialogue = utils.sample_functions_from_graph_and_get_json(
            tool_graph, task, rounds_num
        )
        logger.debug(f"Sampled function list: {function_list}")

        # Generate personas
        personas = utils.get_persona_prompts(
            agents_num, 
            function_dumps_per_dialogue, 
            domain_prompt_dict[domain]
        )
                
        # LangChain Prompt 설정
        pm = PromptMaker(
            agent_num=agents_num,
            rounds_num=rounds_num,
            fewshot=fewshot,
            function_dumps_per_dialogue=function_dumps_per_dialogue,
            domain=domain,
            task=task,
            personas=personas
        )

        logger.debug(f"function_dumps_per_dialogue: {function_dumps_per_dialogue}")

        # LangChain 파이프라인 생성
        main_graph = make_agent_pipeline(pm)

        # utils.draw_langgraph(main_graph, Path(unique_output_fp).parent)

        data_prompt = pm.data_prompt()
        messages = [HumanMessage(content=data_prompt)]

        rounds_events_list = []
        parameters = []

        is_multi_round = (task == "multi_round")
        actual_rounds = rounds_num if is_multi_round else 1

        prev_virtual_output = None
        for rdx in range(actual_rounds):
            logger.info(f"--- Round {rdx+1} ---")

            current_function_list = function_list[rdx]
            current_function_json = utils.get_functions_from_tool_graph(
                current_function_list, 
                json_file_path="src/graph/tool_graph.json"
            )
            current_function_def = json.dumps(
                current_function_json, ensure_ascii=False, indent=4
            )
            logger.debug(f"functions_per_round:\n{current_function_def}")

            # 파라미터 생성
            if prev_virtual_output:  # if it is not the first round
                current_parameters = utils.gen_parameter_values(
                    functions=current_function_def,
                    domain=domain,
                    conversation_history=rounds_events_list[rdx-1] if rdx > 0 else None,
                    prev_function=prev_virtual_output["prev_function"],
                    prev_parameter=prev_virtual_output["prev_parameter"],
                    prev_virtual_output=prev_virtual_output["prev_virtual_output"],
                )
                logger.debug(f"prev_function: {prev_virtual_output['prev_function']}")
                logger.debug(f"prev_virtual_output: {prev_virtual_output['prev_virtual_output']}")
            else:  # if it is the first round
                current_parameters = utils.gen_parameter_values(
                    functions=current_function_def,
                    domain=domain
                )
                
            parameters.append(current_parameters)

            current_prompt = utils.system_prompt_per_round(
                current_function_def, 
                current_parameters
            )
            messages.append(HumanMessage(content=current_prompt))

            events_gen = main_graph.stream(
                {"messages": messages},
                {"recursion_limit": 40},
            )
            rounds_events = list(events_gen)
            rounds_events_list.append(rounds_events)

            virtual_output = None
            if is_multi_round and rdx < actual_rounds - 1:
                current_function_str = current_function_def
                current_parameter_value_str = current_parameters

                virtual_output = utils.virtual_function_call(
                    current_function_str, 
                    current_parameter_value_str
                )
                logger.debug(f"virtual_output:\n{virtual_output}")

                prev_virtual_output = {
                    "prev_function": current_function_str,
                    "prev_parameter": current_parameter_value_str,
                    "prev_virtual_output": virtual_output,
                }

                messages.append(
                    AIMessage(
                        content=(
                            f"[Return Value from Round {rdx+1}' function call]: {virtual_output}\n"
                            "You should use the return value: {virtual_output} to progress the next round conversation "
                            "so that the return value can be used as a parameter for the next function call."
                        ),
                        name="AI_Assistant",
                    )
                )

        conversation_dict = utils.create_conversation_dict(
            dataset_index=i,
            rounds_events_list=rounds_events_list,
            personas=personas,
            domain=domain,
            functions=function_list,
            parameters=parameters,
            task=task,
            turn_num=10,
            round_num=rounds_num
        )

        all_conversations.append(conversation_dict)
        logger.info(f"=== [Dataset {i+1}] Simulation End ===\n")

    # 결과 저장
    with open(f"{unique_output_fp}", "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved all conversations to {unique_output_fp}")


if __name__ == "__main__":
    main()