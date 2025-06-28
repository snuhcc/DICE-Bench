import json
import time                    
import random
import click
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.errors import GraphRecursionError

from src.agent.base import make_agent_pipeline
from src.prompt.base import PromptMaker
from src.utils import utils
from dotenv import load_dotenv
from src.prompt.domain_prompt import domain_prompt_dict
from src.utils.few_shot_params import few_shot_params

import logging
from colorlog import ColoredFormatter
from pathlib import Path

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
                                          

# Load .env from project root (two levels up from this file)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

@click.command()

@click.option("--agent_num", type=int, required=True, help="Number of agents in the dialogue.")
@click.option("--rounds_num", type=int, required=True, help="Number of rounds (use 1 for single_round).")
@click.option("--fewshot", default="", show_default=True, help="Few-shot prompt identifier or content.")
@click.option("--domain", required=True, help="Dialogue domain.")
@click.option("--output_path", required=True, help="Path to the output JSON file.")
@click.option("--task", required=True, type=click.Choice(["single_round", "multi_round"]), help="Task type.")
@click.option("--dataset_num", type=int, default=1, show_default=True, help="Number of dialogues to generate.")
@click.option("--max_turns", type=int, default=20, show_default=True, help="Maximum turns per round.")
def main(agent_num, rounds_num, fewshot, domain, output_path, task, dataset_num, max_turns):

    # All parameters are already available via function arguments; no need for redundant reassignments
    
    # Map shorthand domain names to canonical keys accepted by domain_prompt_dict
    domain_aliases = {
        "Persuassion": "Persuassion_Deliberation_and_Negotiation",
        "Negotiation": "Persuassion_Deliberation_and_Negotiation",
        "Deliberation": "Persuassion_Deliberation_and_Negotiation",
        "Information-Seeking": "Inquiry_and_Information_Seeking",
        "Inquiry": "Inquiry_and_Information_Seeking",
    }

    if domain in domain_aliases:
        logger.debug(f"Mapping domain '{domain}' to canonical '{domain_aliases[domain]}'")
        domain = domain_aliases[domain]

    logger.info(
        "\n🎉 Hyperparameters loaded from CLI:\n"
        f"  - 👥 agent_num: {agent_num}\n"
        f"  - 🛞 max_turns: {max_turns}\n"
        f"  - 🔁 rounds_num: {rounds_num}\n"
        f"  - 🌐 domain: {domain}\n"
        f"  - 💾 output_path: {output_path}\n"
        f"  - 🎯 task: {task}\n"
        f"  - 📊 dataset_num: {dataset_num}\n"
    )
    
    # Determine the path to the tool graph JSON relative to this file
    graph_path = Path(__file__).resolve().parent / "graph" / "tool_graph.json"
    graph_path = str(graph_path)

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        max_idx = len(existing_data)
    except FileNotFoundError:
        print(f"{output_path} is not found.")
        existing_data = []
        max_idx = 0

    with open(graph_path, "r", encoding="utf-8") as f:
        tool_graph = json.load(f)

    for i in range(max_idx, max_idx + dataset_num):
        logger.info(f"=== [Dataset {i+1}] Simulation Start ===")

        function_list, function_dumps_per_dialogue = utils.sample_functions_from_graph_and_get_json(
            tool_graph, task, rounds_num, graph_path
        )
        logger.info(f"--- {function_list}")
        logger.debug(f"Sampled function list: {function_list}")

        personas = utils.gen_persona_prompts(
            agent_num, 
            function_dumps_per_dialogue, 
            domain_prompt_dict[domain]
        )
        
        pm = PromptMaker(
            agent_num=agent_num,
            rounds_num=rounds_num,
            fewshot=fewshot,
            function_dumps_per_dialogue=function_dumps_per_dialogue,
            domain=domain,
            task=task,
            personas=personas,
            max_turns=max_turns
        )

        logger.debug(f"function_dumps_per_dialogue: {function_dumps_per_dialogue}")

        main_graph = make_agent_pipeline(pm)
        
        data_prompt = pm.data_prompt()
        messages = [HumanMessage(content=data_prompt)]

        rounds_events_list = []
        params_ret_val = []

        is_multi_round = (task == "multi_round")
        actual_rounds = rounds_num if is_multi_round else 1

        prev_virtual_output = None

        for rdx in range(actual_rounds):
            round_start_time = time.time()
            
            max_retry = 3
            for attempt in range(max_retry):
                try:
                    current_function_list = function_list[rdx]
                    logger.info(f"--- Round {rdx+1} : {function_list[rdx]} ---")
                    
                    current_function_json = utils.get_functions_from_tool_graph(
                        current_function_list, 
                        json_file_path=graph_path
                    )
                    current_function_def = json.dumps(
                        current_function_json, ensure_ascii=False, indent=4
                    )
                    logger.debug(f"functions_per_round:\n{current_function_def}")

                    first_example, second_example = random.sample(few_shot_params, 2)

                    if prev_virtual_output:
                        current_parameters = utils.gen_parameter_values(
                            functions=current_function_def,
                            domain=domain,
                            first_example=first_example,
                            second_example=second_example,
                            conversation_history=rounds_events_list[rdx-1] if rdx > 0 else None,
                            prev_function=prev_virtual_output["prev_function"],
                            prev_parameter=prev_virtual_output["prev_parameter"],
                            prev_virtual_output=prev_virtual_output["prev_virtual_output"],
                        )
                    else:
                        current_parameters = utils.gen_parameter_values(
                            functions=current_function_def,
                            domain=domain,
                            first_example=first_example,
                            second_example=second_example
                        )

                    current_prompt = utils.system_prompt_per_round(
                        current_function_def, 
                        current_parameters
                    )

                    temp_msg = ""
                    if rdx > 0:
                        prev_conversation = utils.extract_agent_msg([rounds_events], rdx)
                        temp_msg = HumanMessage(
                            content=(
                                f"""
                                    
        Previous conversation: {prev_conversation[0]}

        You should use the above previous conversation and a return value from AI Assistant to progress the next round conversation.

        Make sure the conversation flows naturally from the previous conversation.
                                """
                            ),
                            name="orchestrator"
                        )

                                                 
                    prev_function_def = HumanMessage(content=f"""
        {current_prompt}
        You must continue the conversation for at least {max_turns} turns among the agents. 
        **Do not include the information of both parameter values and function name in one utterance.**
        (e.g. "AI, please find a Thai restaurant in San Francisco." has both parameter values and function name which are find_restaurant and San Francisco and Thai. -> This is wrong.)
        When agent call an AI Assistant, you should not include both the information of parameter values and function name in one utterance.
        **You must call AI Assistant to perform the function, for example, "AI, please find a restaurant in the city."**
        **Must strictly adhere to the following characteristics:**
        {domain_prompt_dict[domain]}
                    """)
                    
                    events_gen = main_graph.stream(
                        {"messages": messages + [temp_msg] + [prev_function_def]},
                        {"recursion_limit": 100},
                    )
                    rounds_events = list(events_gen)

                    reasoning_func_params = utils.gen_reasoning(domain, current_parameters, rounds_events_list)

                    virtual_output_dict = utils.gen_virtual_output(
                        current_function_def,
                        current_parameters
                    )

                    try:
                        reasoning_dict = utils.print_processed_strings([reasoning_func_params])[0]
                        # Merge virtual return values
                        for key, value in virtual_output_dict.items():
                            if key == "returned_nl":
                                continue
                            reasoning_dict.setdefault("return_value", {})[key] = value
                        reasoning_dict["returned_nl"] = virtual_output_dict["returned_nl"]

                        # --- Remove reasoning (2nd list element) so only pure values remain ---
                        # function: [value, reasoning] -> value
                        if isinstance(reasoning_dict.get("function"), list):
                            reasoning_dict["function"] = reasoning_dict["function"][0]

                        # domain: [value, reasoning] -> value
                        if isinstance(reasoning_dict.get("domain"), list):
                            reasoning_dict["domain"] = reasoning_dict["domain"][0]

                        # parameters: {name: [value, reasoning]} -> {name: value}
                        params_val = reasoning_dict.get("parameters")
                        if isinstance(params_val, dict):
                            reasoning_dict["parameters"] = {k: (v[0] if isinstance(v, list) else v) for k, v in params_val.items()}

                        reasoning_func_params = json.dumps(reasoning_dict, ensure_ascii=False)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Error decoding reasoning_func_params: {e}")

                                                  
                    rounds_events.append(
                        {
                            "AI Assistant": {
                                "messages": [
                                    AIMessage(content=virtual_output_dict["returned_nl"], name="AI Assistant")
                                ]
                            }
                        }
                    )

                    if is_multi_round and rdx < actual_rounds - 1:
                        prev_virtual_output = {
                            "prev_function": current_function_def,
                            "prev_parameter": current_parameters,
                            "prev_virtual_output": virtual_output_dict["returned_nl"],
                        }
                        
                                      
                    break

                except GraphRecursionError:
                                                    
                    logger.warning(f"GraphRecursionError occurred. Retrying... ({attempt+1}/{max_retry})")
                    if attempt == max_retry - 1:
                        logger.error("Max retry limit reached. Exiting...")
                        break

                                                      
            params_ret_val.append(reasoning_func_params)
                                  
            rounds_events_list.append(rounds_events)

                                 
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time
            turn_len_per_round = 0
            for round_d in utils.extract_agent_msg([rounds_events]):
                for round_name, msgs in round_d.items():
                    turn_len_per_round += len(msgs)
            logger.info(f"--- Round {rdx+1} took {round_duration:.2f} seconds --- {turn_len_per_round} turns")
        
                                                     
        conversation_dict = utils.create_conversation_dict(
            dataset_index=i,
            rounds_events_list=rounds_events_list,
            personas=personas,
            domain=domain,
            functions=function_list,
            params_ret_val=params_ret_val,
            task=task,
            round_num=rounds_num,
            agent_num=agent_num
        )

                                                        
        existing_data.append(conversation_dict)
        logger.info(f"=== [Dataset {i+1}] Simulation End ===\n")

                                               
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved [Dataset {i+1}] conversations to {output_path}")


if __name__ == "__main__":
    main()