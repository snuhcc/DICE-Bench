import json
import os
from openai import OpenAI

def get_parameter_values(functions):
    client = OpenAI()
    prompt = f"""
        Below is an example of functions and their parameters:
        
        \"functions\": [
        {{
            "function": "get_weather",
            "desc": "Retrieve weather information for a specific location and date.",
            "parameters": [
            {{
                "name": "location",
                "type": "string",
                "desc": "The city or region to check weather for."
            }},
            {{
                "name": "date",
                "type": "string",
                "desc": "Date in MM-DD format."
            }}
            ],
            "return": {{
            "location": "string",
            "date": "string",
            "forecast": "string"
            }}
        }},
        {{
            "function": "book_hotel",
            "desc": "Book a specific hotel on a given date and location.",
            "parameters": [
            {{
                "name": "hotel_name",
                "type": "string",
                "desc": "Name of the hotel."
            }},
            {{
                "name": "date",
                "type": "string",
                "desc": "Date in MM-DD format."
            }},
            {{
                "name": "location",
                "type": "string",
                "desc": "Location of the hotel."
            }}
            ],
            "return": {{
            "hotel_name": "string",
            "location": "string",
            "check_in_date": "string",
            "hotel_booking_confirmation": "string"
            }}
        }}
        ],
        
        \"parameters\": [
        {{'function': "get_weather", 'parameters': {{"location": "Thailand", "date": "06-07"}}}},
        {{'function': "book_hotel", 'parameters': {{"hotel_name": "Hilton Bangkok", "date": "06-07", "location": "Thailand"}}}}
        ],
        
        The above shows how you can structure functions and their parameters, along with example values.
        
        In the same format, please provide example parameters for each function described below:
        
        Example output format:
        [
            {{
                "function": "<function_name>",
                "parameters": {{<parameter_name_1>: <value_1>, <parameter_name_2>: <value_2>, ...}}
            }},
            {{
                "function": "<function2_name>",
                "parameters": {{<parameter_name_1>: <value_1>, <parameter_name_2>: <value_2>, ...}}
            }}
        ]
        
        These are the functions for which you need to generate parameter values:
        {{functions}}
        
        Please specify example parameter values accordingly.
        """
        
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt.replace("{functions}", functions)
            }
        ],
        temperature=0.0,
    )
    
    return completion.choices[0].message.content

def save_data(events_list, save_path=None):
    save_dicts = {}
    metadata_dicts = {}
    for i, events in enumerate(events_list):
        print(f"#{i}th data\n")
        save_list = []
        metadata_list = []
        for event in events:
            if isinstance(event, dict):
                # event는 노드 이름(예: "mediator", "agent_a")을 키로 가지는 딕셔너리
                for node_name, node_data in event.items():
                    # node_data 내부에 messages 키가 있으면 메시지를 꺼낸다
                    if "messages" in node_data:
                        messages = node_data["messages"]
                        # messages는 보통 [AIMessage(...), ...] 같은 리스트
                        for msg in messages:
                            # msg.content만 출력
                            print(f"[{node_name}] {msg.content}")
                            if node_name != "orchestrator":
                                save_list.append({
                                    "name": node_name,
                                    "content": msg.content
                                })
                            else:
                                metadata_list.append({
                                    "content": msg.content
                                })
            print("-" * 10)

        save_dicts[f"data{i}"] = save_list
        metadata_dicts[f"data{i}"] = metadata_list
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_dicts, f, ensure_ascii=False)
        # json.dumps(save_dicts, indent=4)

    return save_dicts, metadata_dicts


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
    
def get_functions_from_tool_graph(tool_list, json_file_path='tool_graph.json'):
    """
    tool_list: ['get_weather', 'book_hotel', ...] 처럼 함수 이름 문자열 목록
    json_file_path: tool_graph.json 파일 경로
    return: tool_list에 포함된 함수들만 골라서 반환하는 dict 예시
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    functions_map = {node["function"]: node for node in data["nodes"]}

    # tool_list에 포함된 함수 이름만 필터링
    filtered = []
    for func_name in tool_list:
        if func_name in functions_map:
            filtered.append(functions_map[func_name])
            
    result = {
        "functions": filtered
    }

    return result
