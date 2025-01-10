import click
import yaml
import json
from langchain_core.messages import HumanMessage
from src.agent.base import make_agent_pipeline
from src.prompt.base import PromptMaker
from src.function.base import BaseFunctionList
from src.utils.utils import save_data
import os
from pathlib import Path


def get_unique_folder_name(folder_path: str) -> str:
    """
    입력받은 folder_path가 이미 존재한다면,
    숫자를 하나씩 붙여가며 존재하지 않는 폴더 경로를 반환.
    """
    if not os.path.exists(folder_path):
        return folder_path

    base_path = folder_path
    counter = 1
    while os.path.exists(folder_path):
        folder_path = f"{base_path}_{counter}"
        counter += 1
    return folder_path


def get_unique_filename(filename):
    """
    파일이 존재한다면, 숫자를 하나씩 붙여가며 존재하지 않는
    파일 경로를 만들어 반환.
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base}{counter}{ext}"
        counter += 1
    return filename

def draw_langgraph(main_graph, save_path):
    from IPython.display import Image, display

    try:
        image_data = main_graph.get_graph(xray=True).draw_mermaid_png()

        # 이미지 파일 경로
        image_path = os.path.join(save_path, "main_graph.png")
        image_path = get_unique_filename(image_path)

        # 이미지 파일 저장
        with open(image_path, "wb") as f:
            f.write(image_data)
        print(f"Image saved as '{image_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")
        pass


@click.command()
@click.option("--agent", default=3, help="Number of agents.")
@click.option("--round", default=1, help="Number of rounds")
@click.option("--fewshot", default="", help="fewshot use")
@click.option("--iter", default=1, help="number of iteration")
@click.option("--domain", default="Persuasion", help="domain")
@click.option("--output_path", default="outputs/test.json", help="Output_path")
@click.option("--task", default="S-S", help="Task")
@click.option("--yaml_path", default=None, help="Predefined yaml import")

def main(agent, round, fewshot, iter, domain, output_path, task, yaml_path):
    # YAML 파일이 지정된 경우 설정 불러오기
    if yaml_path is not None:
        with open(yaml_path, encoding="utf-8") as f:
            yaml_data = yaml.full_load(f)
        agent = yaml_data["agent"]
        round = yaml_data["round"]
        fewshot = yaml_data["fewshot"]
        iter = yaml_data["iter"]
        domain = yaml_data["domain"]
        functions = yaml_data["functions"]
        output_path = yaml_data["output_path"]
        task = yaml_data["task"]

    # 출력 경로에서 폴더와 파일명 분리
    output_path = Path(output_path)
    folder_name = "outputs" / output_path.parent
    file_name = output_path.name

    # 폴더 이름 고유화
    unique_folder = get_unique_folder_name(folder_name)
    os.makedirs(unique_folder, exist_ok=True)  # 폴더 생성

    # 파일 이름도 고유화
    unique_output_path = os.path.join(unique_folder, file_name)
    unique_output_path = get_unique_filename(unique_output_path)

    # Few-shot 프롬프트
    fewshot = fewshot

    # 함수 정보
    func = """
        "functions": [
            {
                "function": "get_weather",
                "desc": "Retrieve weather information for a specific location and date.",
                "parameters": [
                    {
                        "name": "location",
                        "type": "string",
                        "desc": "The city or region to check weather for."
                    },
                    {
                        "name": "date",
                        "type": "string",
                        "desc": "Date in MM-DD format."
                    }
                ],
                "return": {
                    "location": "string",
                    "date": "string",
                    "forecast": "string"
                }
            },
            {
                "function": "book_hotel",
                "desc": "Book a specific hotel on a given date and location.",
                "parameters": [
                    {
                        "name": "hotel_name",
                        "type": "string",
                        "desc": "Name of the hotel."
                    },
                    {
                        "name": "date",
                        "type": "string",
                        "desc": "Date in MM-DD format."
                    },
                    {
                        "name": "location",
                        "type": "string",
                        "desc": "Location of the hotel."
                    }
                ],
                "return": {
                    "hotel_name": "string",
                    "location": "string",
                    "check_in_date": "string",
                    "hotel_booking_confirmation": "string"
                }
            }
        ]
    
    """

    parameter_values = """
    {'function': "get_weather", 'parameters': {"location": "Thailand", "date": "06-07"}}
    {'function': "book_hotel", 'parameters': {"hotel_name": "Hilton Bangkok", "date": "06-07", "location": "Thailand"}}
    """

    # LangGraph 파이프라인 생성
    pm = PromptMaker(agent, round, fewshot, func, parameter_values, domain, iter, task)
    main_graph = make_agent_pipeline(pm)

    # LangGraph 그리기
    draw_langgraph(main_graph, unique_folder)

    # 데이터 생성
    events_list = []
    domain_list = list(pm.get_domain()[0])

    for i in range(iter):
        data_prompt = pm.data_prompt()
        print(f"data_prompt: {data_prompt}")
        events = main_graph.stream(
            {"messages": [HumanMessage(content=data_prompt)]},
            {"recursion_limit": 40},
        )

        print(f"events: {events}")

        events_list.append(events)

    # JSON 저장
    output_file = unique_output_path
    save_dicts, metadata_dicts = save_data(events_list, f"{output_file}.json")

    # 메타데이터 저장
    metadata_path = os.path.splitext(output_file)[0] + "_metadata.json"
    metadata = {
        "domain": domain_list,
        "round": round,
        "funclist": func,
        "parameter_values": parameter_values,
        "orchestrator": metadata_dicts,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"Data saved to '{output_file}'")
    print(f"Metadata saved to '{metadata_path}'")


if __name__ == "__main__":
    main()
