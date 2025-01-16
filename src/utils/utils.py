import json
import os
import re
from pathlib import Path
import yaml

# colorlog 설정
import logging
from colorlog import ColoredFormatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 필요에 따라 DEBUG로 변경 가능

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = ColoredFormatter(
    "%(log_color)s[%(levelname)s]%(reset)s %(blue)s%(name)s%(reset)s: %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red'
    }
)

handler.setFormatter(formatter)
logger.addHandler(handler)

from openai import OpenAI
from src.prompt.domain_prompt import domain_prompt_dict
from src.graph.sample_subgraph import ToolGraphSampler


# =============================================================================
# 1) OpenAI 관련 유틸 함수
# =============================================================================

def gen_parameter_values(functions, domain, conversation_history=None, prev_function=None, prev_parameter=None, prev_virtual_output=None):
    logger.debug("Generating parameter values via OpenAI...")  # debug
    prompt = "Please consider the following context information when generating parameters:"
    
    if conversation_history:  # if it is not the first round
        prompt += f"""
            conversation_history: {conversation_history}
            function: {prev_function}
            parameter: {prev_parameter}
        """
    
    prompt = f"""
        domain: {domain}
        domain_desc: {domain_prompt_dict[domain]}
    """
        
    # one-shot example
    prompt += f"""
        Below is a few-shot example of functions and their parameters:
        
        \"functions\": [
            {{
                "function": "find_hotel",
                "desc": "Find a hotel in a specific location with a specified check-in date.",
                "parameters": [
                    {{
                        "name": "location",
                        "type": "string",
                        "desc": "City or region to search for hotels."
                    }},
                    {{
                        "name": "check_in",
                        "type": "string",
                        "desc": "Check-in date in MM-DD format."
                    }}
                ],
                "return": {{
                    "hotel_name": "string"
                }}
            }}
        ],
            {{
            "function": "book_hotel",
            "desc": "Book a specific hotel on a given date and location.",
            "parameters": [
                {{
                "name": "hotel_name",
                "type": "string",
                "desc": "Name of the hotel to book."
                }},
                {{
                "name": "location",
                "type": "string",
                "desc": "City or region of the hotel."
                }},
                {{
                "name": "check_in_date",
                "type": "string",
                "desc": "Check-in date in MM-DD format."
                }},
                {{
                "name": "check_in_time",
                "type": "string",
                "desc": "Approximate check-in time (HH:MM, 24-hour)."
                }}
            }}
            ],
            "return": {{
                "hotel_name": "string",
                "location": "string",
                "check_in_date": "string",
                "check_in_time": "string",
                }}
        ]
        
        \"parameters\": [
            {{'function': "find_hotel", 'parameters': {{"location": "Paris", "check_in": "07-15"}}}},
            {{'function': "book_hotel", 'parameters': {{"hotel_name": "Hotel Le Meurice", "location": "Paris", "check_in_date": "07-15", "check_in_time": "16:30"}}}},

            {{'function': "find_hotel", 'parameters': {{"location": "New York", "check_in": "08-10"}}}},
            {{'function': "book_hotel", 'parameters': {{"hotel_name": "The Plaza Hotel", "location": "New York", "check_in_date": "08-10", "check_in_time": "14:00"}}}},

            {{'function': "find_hotel", 'parameters': {{"location": "Tokyo", "check_in": "09-05"}}}},
            {{'function': "book_hotel", 'parameters': {{"hotel_name": "Tokyo Imperial Hotel", "location": "Tokyo", "check_in_date": "09-05", "check_in_time": "12:45"}}}},

            {{'function': "find_hotel", 'parameters': {{"location": "Rome", "check_in": "10-20"}}}},
            {{'function': "book_hotel", 'parameters': {{"hotel_name": "Rome Cavalieri", "location": "Rome", "check_in_date": "10-20", "check_in_time": "15:00"}}}},

            {{'function': "find_hotel", 'parameters': {{"location": "Dubai", "check_in": "11-25"}}}},
            {{'function': "book_hotel", 'parameters': {{"hotel_name": "Burj Al Arab", "location": "Dubai", "check_in_date": "11-25", "check_in_time": "13:00"}}}},
        ]
        
        Example output format:
        The output format must strictly be in JSON and follow this structure:
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
        Any text outside of this JSON format (such as explanations or additional context) should not be included.
        """

    if conversation_history is not None:
        prompt += f"""
        As you can see in the one-shot example, the virtual output of the previous function: book_hotel is "Hilton Bangkok", and the output is used as the parameter value for the next round's book_hotel function call.
        
        Like the one-shot example, please **strictly adhere** to the following instructions:
        You must use the virtual output: {prev_virtual_output} of the previous round as the input parameter value for the next round's function call.
        """
    
    prompt += f"""
        The following functions are the functions for which you need to generate parameter values:
        {functions}
    
        Please generate diverse, creative and new parameter values for the given function, strictly adhering to the format shown above, without adding any additional context or explanation.
        """
        
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        top_p=0.8,
    )
    
    max_retries, attempt = 3, 0
    
    while attempt < max_retries:
        try:
            result = completion.choices[0].message.content
            break
        except Exception as e:
            logger.logger(f"Attempt {attempt + 1} failed in gen_parameter_values. Retrying...")
            attempt += 1
            if attempt == max_retries:
                raise RuntimeError("Failed to generate valid parameter values after multiple retries.")
        
    logger.debug(f"Generated parameter values: {result}")
    return result


def virtual_function_call(function_to_call, parameter_values):
    logger.debug(f"Simulating function call: {function_to_call} with params: {parameter_values}")
    client = OpenAI()
    prompt = """   
    Simulate the hypothetical output of the following function call:

    Function: {function_to_call}  
    Parameters: {parameter_values}  

    Based on the function and parameter details provided, generate a hypothetical output that aligns with the expected behavior of the function.  
    Only return the hypothetical output, without any additional context or explanation.
    """

    # Replace placeholders with actual values
    prompt = prompt.replace("{function_to_call}", function_to_call)
    prompt = prompt.replace("{parameter_values}", parameter_values)
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a virtual Python runtime environment."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            top_p=0.8,
        )
        result = completion.choices[0].message.content
        logger.debug(f"Virtual function call result: {result}")
        return result

    except client.error.OpenAIError as e:
        logger.error(f"An error occurred in virtual_function_call: {e}")
        return f"An error occurred: {e}"


def get_persona_prompts(agent_num, function_dumps_per_dialogue, domain_desc):
    client = OpenAI()
    persona_generation_prompt = f"""
        Your task is to generate unique and responsible personas for agents participating in a multi-agent conversation system, based on the provided function list: {function_dumps_per_dialogue}.

        **Guidelines for generating the personas:**
        - Ensure each persona has a clear and distinct role, personality traits, and communication style while adhering to ethical standards.
        - Avoid including or reinforcing stereotypes, biases, or potentially offensive traits in the personas.
        - Tailor the personas to contribute effectively to the conversation's goals, maintain balance within the group dynamics, and promote positive and inclusive interactions.
        - Use concise yet descriptive language to outline the persona’s primary focus and approach to the discussion.
        - Avoid repetitive characteristics across different personas to ensure diversity and fairness.
        - Incorporate elements from the provided domain description when generating conversation: \n{domain_desc}.
        - Ensure that all personas align with ethical communication practices and promote a respectful, constructive dialogue.

        **Examples of personas:**
        1. A thoughtful and resourceful problem-solver who likes optimizing plans for the group's benefit...
        2. A detail-oriented and practical thinker who ensures that the plans are realistic and well-organized...
        3. A spontaneous and energetic planner who loves initiating plans and suggesting creative ideas...

        **Response format:**
        Provide the requested number of personas in the following format:
        - **agent_a Persona**: [Description ...]
        - **agent_b Persona**: [Description ...]
        - **agent_c Persona**: [Description ...]
        - (Continue for the specified number of agents.)

        Now, generate {agent_num} personas for the agents in the conversation.
    """
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and ethical assistant."},
            {"role": "user", "content": persona_generation_prompt}
        ],
        temperature=0.7,
        top_p=0.8,
    )
    max_retries, attempt = 3, 0
    while attempt < max_retries:
        try:
            resp = completion.choices[0].message.content
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed in get_persona_prompts. Retrying...")
            attempt += 1
            if attempt == max_retries:
                raise RuntimeError("Failed to generate valid persona prompts after multiple retries.")
    
    pattern = r"- \*\*agent_\w Persona\*\*: (.*?)(?=\n- \*\*agent_\w|\Z)"
    
    persona_prompts = None
    try:
        persona_prompts = re.findall(pattern, resp, flags=re.DOTALL)
        logger.debug(f"Persona prompts extracted: {persona_prompts}")
    except ValueError as e:
        logger.error(f"GPT response format mismatch: {e}")
        logger.error(f"GPT response:\n{resp}")
        return None
    
    return persona_prompts


# =============================================================================
# 2) JSON/대화 파싱 관련 유틸 함수
# =============================================================================

def extract_json(text):
    logger.debug("Extracting JSON block from text...")
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError("JSON section not found in the text.")


def parse_json_functions(text):
    logger.debug("Parsing JSON functions from text...")
    json_data = extract_json(text)
    parameter_values = json.loads(json_data)
    parsed_results = []
    for item in parameter_values:
        function_name = item.get("function")
        parameters = item.get("parameters", {})
        params_string = ", ".join(f"{key}={value}" for key, value in parameters.items())
        parsed_results.append({
            "function_name": function_name,
            "parameters": parameters,
            "formatted": f"{function_name}({params_string})"
        })
    logger.debug(f"Parsed function info: {parsed_results}")
    return parsed_results


def system_prompt_per_round(functions_per_round, parameter_values_per_round):
    logger.debug("Building system prompt per round...")
    system_prompt_per_round = """
        - Make sure the conversation naturally incorporates the function “\n{functions_per_round}\n” and its associated parameter values in a seamless and unforced manner.
        - Engage in a discussion to negotiate the following parameters within the conversation:
            {parameter_values_per_round}
    """

    prompt = system_prompt_per_round.format(
        functions_per_round=functions_per_round,
        parameter_values_per_round=parameter_values_per_round
    )

    return prompt


# =============================================================================
# 3) 파일/폴더/그래프 처리 관련 유틸 함수
# =============================================================================

def create_unique_output_path(output_path: str, task: str) -> str:
    folder_path = Path(output_path)  # "outputs/dialogue/"
    unique_folder_path = get_unique_folder_name(str(folder_path))
    os.makedirs(unique_folder_path, exist_ok=True)
    file_name = f'{task}.json'
    
    unique_output_fp = Path(unique_folder_path) / file_name
    logger.debug(f"Unique output file path created: {unique_output_fp}")
    return unique_output_fp


def get_unique_folder_name(folder_path: str) -> str:
    logger.debug(f"Checking folder existence for: {folder_path}")
    if not os.path.exists(folder_path):
        return folder_path

    base_path = folder_path
    counter = 1
    while os.path.exists(folder_path):
        folder_path = f"{base_path}_{counter}"
        counter += 1
    logger.debug(f"Folder name updated to: {folder_path}")
    return folder_path


def get_unique_filename(filename):
    logger.debug(f"Checking file existence for: {filename}")
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base}{counter}{ext}"
        counter += 1
    logger.debug(f"Unique filename: {filename}")
    return filename


def draw_langgraph(main_graph, save_path):
    from IPython.display import Image, display
    try:
        image_data = main_graph.get_graph(xray=True).draw_mermaid_png()
        image_path = os.path.join(save_path, "main_graph.png")
        image_path = get_unique_filename(image_path)

        with open(image_path, "wb") as f:
            f.write(image_data)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        pass
    

def get_functions_from_tool_graph(tool_list, json_file_path='tool_graph.json'):
    logger.debug(f"Filtering functions from tool_graph: {tool_list}")
    tool_list = [tool_list] if isinstance(tool_list, str) else tool_list
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    functions_map = {node["function"]: node for node in data["nodes"]}

    filtered = []
    for func_name in tool_list:
        if func_name in functions_map:
            filtered.append(functions_map[func_name])
            
    result = {
        "functions": filtered
    }
    logger.debug(f"Filtered function result: {result}")
    return result


def _sample_function_list(graph_sampler, task, rounds_num):
    logger.debug(f"Sampling function list with task={task}, rounds_num={rounds_num}")
    if task == "single_round":
        function_list = graph_sampler.sample_node()
    elif task == "multi_round":
        function_list = graph_sampler.sample_graph(rounds_num=rounds_num)
    else:
        raise ValueError(f"Invalid task: {task}")
    logger.debug(f"Sampled function list: {function_list}")
    return function_list


def sample_functions_from_graph_and_get_json(
    tool_graph: dict, 
    task: str, 
    rounds_num: int, 
    tool_graph_file: str = "src/graph/tool_graph.json"
) -> str:
    graph_sampler = ToolGraphSampler(tool_graph)
    function_list = _sample_function_list(graph_sampler, task, rounds_num)
    function_json = get_functions_from_tool_graph(function_list, tool_graph_file)
    json_str = json.dumps(function_json, ensure_ascii=False, indent=4)
    logger.debug(f"sample_functions_from_graph_and_get_json output:\n{json_str}")
    return function_list, json_str

def load_yaml(yaml_path: str) -> dict:

    if not os.path.exists(yaml_path):
        logger.error(f"YAML file not found: {yaml_path}")
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    required_keys = [
        "agent_num",
        "rounds_num",
        "domain",
        "output_path",
        "task",
        "dataset_num",
    ]
    for rk in required_keys:
        if rk not in yaml_data:
            logger.error(f"Required key is missing in the YAML file: {rk}")
            raise KeyError(f"Required key is missing in the YAML file: {rk}")

    # 필요한 하이퍼파라미터를 변수에 할당
    agents_num = yaml_data["agent_num"]
    rounds_num = yaml_data["rounds_num"]
    domain = yaml_data["domain"]
    output_path = yaml_data["output_path"]
    task = yaml_data["task"]
    fewshot = yaml_data["fewshot"]
    dataset_num = yaml_data["dataset_num"]

    # 이모지 로그
    logger.info(
        "\n🔧 Hyperparameters loaded from YAML:\n"
        f"  - 👥 agents_num: {agents_num}\n"
        f"  - 🔁 rounds_num: {rounds_num}\n"
        f"  - 🌐 domain: {domain}\n"
        f"  - 💾 output_path: {output_path}\n"
        f"  - 🎯 task: {task}\n"
        f"  - 📊 dataset_num: {dataset_num}\n"
    )

    return agents_num, rounds_num, fewshot, domain, output_path, task, dataset_num


# =============================================================================
# 4) 대화 저장용 유틸 함수
# =============================================================================


def print_processed_strings(raw_strings):
    processed_json_list = []  # 유효한 JSON 객체를 저장할 리스트
    logger.debug("Processing list of raw strings...")

    for idx, raw_string in enumerate(raw_strings):
        logger.debug(f"Processing string at index {idx}: {raw_string}")
        match = re.search(r"```json\n(.*?)\n```", raw_string, re.DOTALL)
        if match:
            json_content = match.group(1)
            logger.debug(f"Found JSON block in string at index {idx}.")
            try:
                # JSON 데이터 파싱
                json_object = json.loads(json_content)
                processed_json_list.append(json_object)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON at index {idx}: {e}")
        else:
            logger.debug(f"No JSON block found in string at index {idx}, skipping.")

    return processed_json_list

def create_metadata(idx, personas, domain, functions, parameters, task, turn_num, round_num):
    personas_dict = {}
    for pdx, persona in enumerate(personas):
        agent_key = f"agent_{chr(pdx + 97)}"
        personas_dict[agent_key] = persona
        logger.debug(f"Assigned persona to {agent_key}")

    processed_params = print_processed_strings(parameters)

    metadata = {
        "diag_id": idx,
        "user_personas": personas_dict,
        "functions": functions,
        "parameters": processed_params,
        "category": domain,
        "task": task,
        "turn_num": turn_num,
        "round_num": round_num,
    }
    logger.debug(f"Metadata created: {metadata}")
    return metadata


def create_conversation_dict (
    dataset_index: int,
    rounds_events_list,
    personas,
    domain,
    functions,
    parameters,
    task: str,
    turn_num: int,
    round_num: int
):
    metadata = create_metadata(dataset_index, personas, domain, functions, parameters, task, turn_num, round_num)

    conversation_dict = {
        "metadata": metadata,
        "messages": []
    }

    messages_flat = []
    for round_idx, round_events in enumerate(rounds_events_list, start=1):
        round_dict = {f'Round {round_idx}': []}
        
        for event in round_events:
            if isinstance(event, dict):
                for agent_name, node_data in event.items():
                    if agent_name == "orchestrator":
                        continue
                    if "messages" in node_data:
                        for msg in node_data["messages"]:
                            user_label = agent_name
                            
                            round_dict[f'Round {round_idx}'].append({
                                "speaker": user_label,  
                                "message": msg.content
                            })

        messages_flat.append(round_dict)

    conversation_dict["messages"] = messages_flat
    logger.debug(f"Final conversation_dict: {conversation_dict}")
    return conversation_dict