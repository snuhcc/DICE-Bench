import json
import os
import re
from pathlib import Path
import yaml
import random

from src.colorlog import get_logger

# Unified project-wide logger (colorized)
logger = get_logger(__name__)

from openai import OpenAI
from src.prompt.domain_prompt import domain_prompt_dict
from src.graph.sample_subgraph import ToolGraphSampler


def gen_reasoning(domain, parameters, rounds_events_list) -> str:
    conversation_history = extract_agent_msg(rounds_events_list)
    prompt = f"""
        The format of the output should be the following:
        {{
            "function": ["<function_name>", "<reasoning for why the function is selected given the conversation history>"],
            "parameters": {{
                "<parameter_name1>": ["<parameter_value1>", "<reasoning for why the parameter value is selected given the conversation history>"],
                "<parameter_name2>": ["<parameter_value2>", "<reasoning for why the parameter value is selected given the conversation history>"],
                ...
            }},
            "domain": ["{domain}", "<reasoning for why the conversation is {domain}>"]
        }}
        
        The conversation history is given as follows:
        {conversation_history}
        
        The functions and parameters are given as follows:
        {parameters}
        
        Please generate the reasoning for why the parameter values, functions, and domain are selected given the conversation history.
    """
    client = OpenAI()
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "Produce valid JSON describing function & parameters & domain selection reasoning."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            result = completion.choices[0].message.content
            logger.debug(f"[gen_reasoning] Attempt {attempt} raw:\n{result}")

            processed = print_processed_strings([result])
            if processed:
                logger.debug("[gen_reasoning] Valid JSON parsed successfully.")
                return result

            logger.warning(
                f"[gen_reasoning] Attempt {attempt} JSON parse failed. Retrying..."
            )

        except Exception as e:
            logger.warning(f"[gen_reasoning] Attempt {attempt} error: {e}. Retrying...")

    raise RuntimeError(
        "[gen_reasoning] Could not produce valid JSON after multiple retries."
    )


def gen_parameter_values(
    functions,
    domain,
    first_example,
    second_example,
    conversation_history=None,
    prev_function=None,
    prev_parameter=None,
    prev_virtual_output=None,
):
    logger.debug("Generating parameter values via OpenAI...")

    domain_desc_part = f"""
domain: {domain}
domain_desc: {domain_prompt_dict[domain]}
    """

    examples_part = f"""
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
    }},
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
        ],
        "return": {{
            "hotel_name": "string",
            "location": "string",
            "check_in_date": "string",
            "check_in_time": "string"
        }}
    }}
]

Below are list of five examples of parameter values for the given function. You only need to generate one example:
# first example
{first_example}

# second example
{second_example}

Example output format:
The output format must strictly be in JSON and follow this structure:
[
    {{
        "function": "<function_name>",
        "parameters": {{ "<parameter_name_1>": "<value_1>", "<parameter_name_2>": "<value_2>", ... }}
    }},
    {{
        "function": "<function2_name>",
        "parameters": {{ "<parameter_name_1>": "<value_1>", "<parameter_name_2>": "<value_2>", ... }}
    }}
]
Any text outside of this JSON format (such as explanations or additional context) should not be included.
    """

    if conversation_history is not None:
        prev_round_part = f"""
As you can see in the one-shot example, the virtual output of the previous function: {prev_function} might be used as the input parameter value for the next round's function call.

You must use the virtual output: {prev_virtual_output} of the previous round as the input parameter value for the next round's function call (if it logically applies).
        """
    else:
        prev_round_part = ""

    final_prompt = f"""
{domain_desc_part}
{examples_part}
{prev_round_part}

The following functions are the functions for which you need to generate parameter values:
{functions}

Please generate diverse and creative parameter values for the given function(s), strictly adhering to the JSON format shown above, without adding any additional context or explanation.
Also, make sure to increase the coherernce between the parameter values being generated.
    """

    client = OpenAI()
    max_retries, attempt = 3, 0
    result = None

    while attempt < max_retries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": final_prompt},
                ],
                temperature=0.8,
                top_p=0.8,
            )
            result = completion.choices[0].message.content
            break
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1} failed in gen_parameter_values: {e}. Retrying..."
            )
            attempt += 1
            if attempt == max_retries:
                raise RuntimeError(
                    "Failed to generate valid parameter values after multiple retries."
                )

    logger.debug(f"Generated parameter values: {result}")
    return result


def gen_virtual_output(function_to_call, parameter_values):
    logger.debug(
        f"Simulating function call: {function_to_call} with params: {parameter_values}"
    )
    client = OpenAI()
    prompt = f"""
Simulate the hypothetical output of the following function call:

Function: {function_to_call}
Parameters: {parameter_values}

You are a voice assistant responding naturally with the final result of this function call. You need to return both the short and concise return value of the function call, and the natural language response of the function call.
**Important**: 
- Do not mention that this is a simulation or hypothetical. 
- Return only a single, direct response in a natural language as if the function actually executed successfully.
- Keep it concise and natural, like a single short paragraph.

The format of the output should be the following:
{{
    "<returned_value1>": "<short and concise return value of the function call>"
    "<returned_value2>": "<short and concise return value of the function call>"
    ...
    "returned_nl": "<natural language response of the function call given the return values>"
}}
"""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a virtual Python runtime environment.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                top_p=0.8,
            )
            result = completion.choices[0].message.content
            logger.debug(f"[Attempt {attempt}] Raw function call result:\n{result}")

            processed = print_processed_strings([result])

            if processed:
                if "returned_nl" in processed[0]:
                    logger.debug("[gen_virtual_output] Valid JSON parsed successfully.")
                    return processed[0]
                else:
                    logger.warning(
                        f"[Attempt {attempt}] JSON parse failed. Retrying..."
                    )

            logger.warning(f"[Attempt {attempt}] JSON parse failed. Retrying...")

        except Exception as e:
            logger.warning(f"[Attempt {attempt}] Error occurred: {e}. Retrying...")

    raise RuntimeError(
        "[gen_virtual_output] Could not produce valid JSON after multiple retries."
    )


def gen_persona_prompts(agent_num, function_dumps_per_dialogue, domain_desc):
    backup_personas = [
        "A cooperative agent who values clarity and thoughtful inquiry.",
        "An empathetic agent who offers constructive feedback and support.",
        "An analytical agent who checks facts and reasoning carefully.",
        "A creative agent who suggests new perspectives without bias.",
        "A calm, solution-focused agent who respects fair communication.",
        "A detail-oriented agent who avoids misinformation or stereotypes.",
        "A polite, curious agent open to diverse viewpoints and data.",
        "An agent fostering collaboration, ensuring all voices are included.",
        "An agent emphasizing evidence-based reasoning and honest discussion.",
        "An agent promoting respectful discourse, mindful of sensitivities.",
    ]

    persona_generation_prompt = f"""
Your task is to generate concise, unique and responsible personas for agents participating 
in a multi-agent conversation system, based on the provided function list: {function_dumps_per_dialogue}.

**Guidelines**:
- Ensure each persona has a clear and distinct role, personality traits, and communication style while adhering to ethical standards.
- Avoid reinforcing stereotypes, biases, or offensive traits.
- Tailor the personas to contribute effectively to the conversation's goals and maintain balance.
- Use concise yet descriptive language.
- Avoid repetitive characteristics across different personas to ensure diversity and fairness.
- Incorporate elements from the provided domain description when generating conversation: {domain_desc}.
- Ensure all personas align with ethical communication practices.
- Generate personas in two sentences.

**Examples**:
1. A thoughtful and resourceful problem-solver ...
2. A detail-oriented and practical thinker ...
3. A spontaneous and energetic planner ...

**Response format**:
- **agent_a Persona**: [Description ...]
- **agent_b Persona**: [Description ...]
- ...

Generate {agent_num} personas for the agents in the conversation.
"""

    client = OpenAI()
    max_retries = 3
    attempt = 0
    resp = None
    final_personas = []

    pattern = r"- \*\*agent_\w Persona\*\*: (.*?)(?=\n- \*\*agent_\w|\Z)"

    while attempt < max_retries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful and ethical assistant.",
                    },
                    {"role": "user", "content": persona_generation_prompt},
                ],
                temperature=0.8,
                top_p=0.8,
            )
            resp = completion.choices[0].message.content
            logger.debug(f"GPT raw response:\n{resp}")

            persona_prompts = re.findall(pattern, resp, flags=re.DOTALL)
            persona_prompts = [p.strip() for p in persona_prompts]
            logger.debug(f"Persona prompts extracted: {persona_prompts}")

            if len(persona_prompts) >= agent_num:
                final_personas = persona_prompts[:agent_num]
                break
            else:
                logger.warning(
                    f"GPT response extracted {len(persona_prompts)} Personas, which is less than the required {agent_num}. Retrying..."
                )
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed in gen_persona_prompts: {e}")

        attempt += 1

    if len(final_personas) < agent_num:
        shortfall = agent_num - len(final_personas)
        logger.warning(
            f"재시도 후에도 Persona가 {shortfall}개 부족. 백업용 Persona로 채웁니다."
        )

        if shortfall > len(backup_personas):
            raise ValueError(
                f"백업용 Persona도 부족합니다. 필요한 개수: {shortfall}, 백업용: {len(backup_personas)}"
            )
        fill_personas = random.sample(backup_personas, shortfall)
        final_personas.extend(fill_personas)

    return final_personas


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
        parsed_results.append(
            {
                "function_name": function_name,
                "parameters": parameters,
                "formatted": f"{function_name}({params_string})",
            }
        )
    logger.debug(f"Parsed function info: {parsed_results}")
    return parsed_results


def system_prompt_per_round(functions_per_round, parameter_values_per_round):
    logger.debug("Building system prompt per round...")
    system_prompt_per_round = """
        - Make sure the conversation naturally incorporates the function \n{functions_per_round} 
          and its associated parameter values in a seamless and unforced manner.
        - Engage in a discussion to negotiate the following parameters within the conversation:
            {parameter_values_per_round}
    """

    prompt = system_prompt_per_round.format(
        functions_per_round=functions_per_round,
        parameter_values_per_round=parameter_values_per_round,
    )
    return prompt


def create_unique_output_path(output_path: str, task: str) -> str:
    folder_path = Path(output_path)
    unique_folder_path = get_unique_folder_name(str(folder_path))
    os.makedirs(unique_folder_path, exist_ok=True)
    file_name = f"{task}.json"
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

    try:
        image_data = main_graph.get_graph(xray=True).draw_mermaid_png()
        image_path = os.path.join(save_path, "main_graph.png")
        image_path = get_unique_filename(image_path)

        with open(image_path, "wb") as f:
            f.write(image_data)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        pass


def get_functions_from_tool_graph(tool_list, json_file_path="tool_graph.json"):
    logger.debug(f"Filtering functions from tool_graph: {tool_list}")
    tool_list = [tool_list] if isinstance(tool_list, str) else tool_list

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    functions_map = {node["function"]: node for node in data["nodes"]}
    filtered = []
    for func_name in tool_list:
        if func_name in functions_map:
            filtered.append(functions_map[func_name])

    result = {"functions": filtered}
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
    tool_graph_file: str = str(Path(__file__).resolve().parent.parent / "graph" / "tool_graph.json"),
) -> (list, str):
    graph_sampler = ToolGraphSampler(tool_graph)
    function_list = _sample_function_list(graph_sampler, task, rounds_num)
    function_json = get_functions_from_tool_graph(function_list, tool_graph_file)
    json_str = json.dumps(function_json, ensure_ascii=False, indent=4)
    logger.debug(f"sample_functions_from_graph_and_get_json output:\n{json_str}")
    return function_list, json_str


def load_yaml(yaml_path: str) -> tuple:
    if not os.path.exists(yaml_path):
        logger.error(f"YAML file not found: {yaml_path}")
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    required_keys = [
        "agent_num",
        "rounds_num",
        "fewshot",
        "domain",
        "output_path",
        "task",
        "dataset_num",
        "max_turns",
    ]
    for rk in required_keys:
        if rk not in yaml_data:
            logger.error(f"Required key is missing in the YAML file: {rk}")
            raise KeyError(f"Required key is missing in the YAML file: {rk}")

    agents_num = yaml_data["agent_num"]
    rounds_num = yaml_data["rounds_num"]
    fewshot = yaml_data["fewshot"]
    domain = yaml_data["domain"]
    output_path = yaml_data["output_path"]
    task = yaml_data["task"]
    dataset_num = yaml_data["dataset_num"]
    max_turns = yaml_data["max_turns"]

    return (
        agents_num,
        rounds_num,
        fewshot,
        domain,
        output_path,
        task,
        dataset_num,
        max_turns,
    )


def print_processed_strings(raw_strings):
    processed_json_list = []
    for idx, raw_string in enumerate(raw_strings):
        logger.debug(f"Attempting to process string at index {idx}")
        match = re.search(r"```json\n(.*?)```", raw_string, re.DOTALL)

        if match:
            json_content = match.group(1)
            try:
                json_object = json.loads(json_content)
                processed_json_list.append(json_object)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Invalid JSON at index {idx}: {e}. Content: {json_content}"
                )
        else:
            try:
                if isinstance(raw_string, str):
                    parsed = json.loads(raw_string)
                else:
                    parsed = raw_string
                processed_json_list.append(parsed)
            except:
                logger.debug(f"No valid JSON found at index {idx}, skipping.")
    return processed_json_list


def create_metadata(
    idx, personas, domain, functions, params_ret_val, task, round_num, agent_num
):
    personas_dict = {}
    for pdx, persona in enumerate(personas):
        agent_key = f"agent_{chr(pdx + 97)}"
        personas_dict[agent_key] = persona

    processed_params = print_processed_strings(params_ret_val)

    def _clean_entry(entry):
        """Remove the reasoning (2nd list element) from function, parameters and domain."""
        if not isinstance(entry, dict):
            return entry
        cleaned = entry.copy()

        # function: [value, reasoning] -> value
        func_val = cleaned.get("function")
        if isinstance(func_val, list) and func_val:
            cleaned["function"] = func_val[0]

        # domain: [value, reasoning] -> value
        domain_val = cleaned.get("domain")
        if isinstance(domain_val, list) and domain_val:
            cleaned["domain"] = domain_val[0]

        # parameters: {name: [value, reasoning]} -> {name: value}
        params_val = cleaned.get("parameters")
        if isinstance(params_val, dict):
            cleaned["parameters"] = {
                k: (v[0] if isinstance(v, list) and v else v) for k, v in params_val.items()
            }

        return cleaned

    processed_params = [_clean_entry(e) for e in processed_params]

    metadata = {
        "diag_id": idx,
        "user_personas": personas_dict,
        "functions": functions,
        "params_ret_val": processed_params,
        "category": domain,
        "task": task,
        "round_num": round_num,
        "agent_num": agent_num,
    }
    return metadata


def get_summary(dialogue):
    client = OpenAI()
    summary_prompt = f"""
        Please summarize the following dialogue:
        {dialogue}
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": summary_prompt},
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content


def extract_agent_msg(rounds_events_list, round_idx=1):
    messages_flat = []
    for rnd_idx, round_events in enumerate(rounds_events_list, start=round_idx):
        round_dict = {f"Round {rnd_idx}": []}
        for event in round_events:
            if isinstance(event, dict):
                for agent_name, node_data in event.items():
                    if agent_name == "orchestrator":
                        continue
                    if "messages" in node_data:
                        for msg in node_data["messages"]:
                            user_label = agent_name
                            round_dict[f"Round {rnd_idx}"].append(
                                {"speaker": user_label, "message": msg.content}
                            )
        messages_flat.append(round_dict)
    return messages_flat


def create_conversation_dict(
    dataset_index: int,
    rounds_events_list,
    personas,
    domain,
    functions,
    params_ret_val,
    task: str,
    round_num: int,
    agent_num: int,
):
    metadata = create_metadata(
        idx=dataset_index,
        personas=personas,
        domain=domain,
        functions=functions,
        params_ret_val=params_ret_val,
        task=task,
        round_num=round_num,
        agent_num=agent_num,
    )

    conversation_dict = {"metadata": metadata, "messages": []}

    messages_flat = extract_agent_msg(rounds_events_list)
    conversation_dict["messages"] = messages_flat

    logger.debug(f"Final conversation_dict: {conversation_dict}")
    return conversation_dict
