import os
import json
import yaml
from pathlib import Path
import pprint
import re

import click
from langchain_core.messages import HumanMessage

# 내부 라이브러리 임포트
from src.agent.base import make_agent_pipeline
from src.prompt.base import PromptMaker
from src.function.base import BaseFunctionList
from src.utils import utils
from src.graph.sample_subgraph import ToolGraphSampler


@click.command()
@click.option("--yaml_path", default=None, help="Path to a predefined YAML file.")
def main(
    yaml_path,
):

    # 1. YAML 파일이 지정된 경우: 해당 파일에서 설정 읽어 오기
    if yaml_path is not None:
        with open(yaml_path, encoding="utf-8") as f:
            yaml_data = yaml.full_load(f)

        # YAML 데이터에 있는 설정을 우선적으로 사용
        agents_num = yaml_data["agent_num"]
        rounds_num = yaml_data["rounds_num"]
        fewshot = yaml_data["fewshot"]
        domain = yaml_data["domain"]
        output_path = yaml_data["output_path"]
        task = yaml_data["task"]
        dataset_num = yaml_data["dataset_num"]
    else:
        # YAML 없는 경우, 기본값 또는 CLI 옵션 사용
        functions = None

    # 2. 출력 경로를 기반으로 폴더와 파일명 분리
    output_path_obj = Path(output_path)
    folder_path = output_path_obj.parent  # 예: "outputs"
    file_name = output_path_obj.name      # 예: "test.json"

    # 3. 폴더 이름 고유화 & 폴더 생성
    unique_folder_path = Path(utils.get_unique_folder_name(folder_path))
    os.makedirs(unique_folder_path, exist_ok=True)

    # 4. 파일 이름도 고유화
    unique_output_file = utils.get_unique_filename(str(unique_folder_path / file_name))
    
    with open('src/graph/tool_graph.json', 'r') as f:
        tool_graph = json.load(f)

    graph_sampler = ToolGraphSampler(tool_graph)
    
    function_list = []
    for i in range(dataset_num):
        if task == 'S-S':
            func = graph_sampler.sample_node()
            function_list.append(func)
        elif task == 'S-M':
            func = graph_sampler.sample_undirected_path(num_nodes=2)
            function_list.append(func)
        elif task == 'M-S':
            func = graph_sampler.sample_directed_path(num_nodes=2)
            function_list.append(func)
        elif task == 'M-M':
            level_functions, edges = graph_sampler.sample_tree(num_levels=2, nodes_per_level=2)
            function_list.extend((level_functions, edges))
        else:
            raise ValueError(f"Invalid task: {task}")
        
    
    # 6. 함수 파라미터 예시(실행 시 활용할 수 있는 값)

    print(f'function_list: {function_list}')

    for i in range(dataset_num):
        function_json = utils.get_functions_from_tool_graph(function_list[i], json_file_path='src/graph/tool_graph.json')
        
        # generate functions for dialogue
        functions_per_dialogue = json.dumps(function_json, ensure_ascii=False, indent=4)
        
        # generate paramters
        parameter_values = utils.get_parameter_values(functions_per_dialogue)
        
        # generate personas
        personas = utils.get_personas(domain, functions_per_dialogue, parameter_values, persona_num=agents_num)
        
        print(f'personas: {personas}')
        
        # 함수 정보 추출
        # parsed_details = utils.pars_json_functions(parameter_values)
        # print(f'\nparsed_details: {parsed_details}\n')
        # print(f'\ntype(parsed_details): {type(parsed_details)}\n')  
        
        # 7. LangChain Prompt 설정
        pm = PromptMaker(
            agent_num=agents_num,
            rounds_num=rounds_num,
            fewshot=fewshot,
            functions_per_dialogue=functions_per_dialogue,
            parameter_values=parameter_values,
            domain=domain,
            task=task,
        )

        # 8. LangChain 파이프라인 생성
        main_graph = make_agent_pipeline(pm)

        # 9. 파이프라인 구조 시각화 후 저장
        utils.draw_langgraph(main_graph, unique_folder_path)

        # 10. 결과 이벤트 목록 생성
        events_list = []

        data_prompt = pm.data_prompt()

        # main_graph.stream 호출
        events = main_graph.stream(
            {"messages": [HumanMessage(content=data_prompt)]},
            {"recursion_limit": 40},
        )

        events_list.append(events)

    # # 11. 결과 JSON 저장
    # save_dicts, metadata_dicts = utils.save_data(events_list, f"{unique_output_file}.json")

    # # 12. 메타데이터 저장
    # metadata_path = os.path.splitext(unique_output_file)[0] + "_metadata.json"
    # metadata = {
    #     "domain": domain_list,
    #     "round": rounds_num,
    #     "funclist": func,
    #     "parameter_values": parameter_values,
    #     "orchestrator": metadata_dicts,
    # }
    # with open(metadata_path, "w", encoding="utf-8") as f:
    #     json.dump(metadata, f, ensure_ascii=False)

    # print(f"Data saved to '{unique_output_file}'")
    # print(f"Metadata saved to '{metadata_path}'")


if __name__ == "__main__":
    main()