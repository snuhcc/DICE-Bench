import json
import logging
from typing import Dict, Any, List
import re
from vllm import LLM

logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    max_tokens: int = 8196,
    tensor_parallel_size: int = 1,
) -> LLM:
    """Load a vLLM model without YAML dependency.

    Parameters
    ----------
    model_name : str
        HF model repo or local path.
    max_tokens : int, default 8196
        Maximum sequence length supported by the model.
    tensor_parallel_size : int, default 1
        vLLM tensor parallel degree.
    """

    logger.info(
        f"Loading model: {model_name} (tensor_parallel_size={tensor_parallel_size}, max_tokens={max_tokens})"
    )

    llm = LLM(
        model=model_name,
        max_model_len=max_tokens,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    logger.info("Model loaded successfully.")
    return llm


def load_function_docs(function_docs_path: str) -> Dict[str, Any]:
    logger.info(f"Loading function docs from: {function_docs_path}")
    with open(function_docs_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    logger.info("Function docs loaded.")
    return docs


def load_dialogue_data(dialogue_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading dialogue data from: {dialogue_path}")
    with open(dialogue_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Dialogue data loaded.")
    return data


def build_few_shot_example() -> str:
    needed_function_example = r"""
[Conversation Example]
User: "I want to schedule a reminder for tomorrow 3 PM to call John."
Assistant: "Sure, I can help with that."

[How the assistant's final JSON result should look like]
{
  "function": [
    "set_reminder",
    "We need to remind the user tomorrow at 3 PM to call John. So 'set_reminder' is appropriate."
  ],
  "parameters": {
    "date": [
      "tomorrow",
      "User wants the reminder for tomorrow."
    ],
    "time": [
      "15:00",
      "Converted 3 PM to 24-hour format => 15:00"
    ],
    "reminder_task": [
      "Call John",
      "User specifically wants to call John."
    ]
  }
}
"""

    no_function_example = r"""
[Conversation Example]
User: "Just wanted to say hello and see how you are doing."
Assistant: "I'm doing well, thanks for asking!"

[How the assistant's final JSON result should look like]
{
  "function": [
    "none",
    "No function is needed because the user is just greeting."
  ],
  "parameters": {}
}
"""

    create_event_example = r"""
[Conversation Example]
User: "We should schedule an interview on March 10th at 3 PM."
Assistant: "Sure, let me set that up."

[How the assistant's final JSON result should look like]
{
  "function": [
    "create_event",
    "The user explicitly requests scheduling an interview on March 10th at 3 PM, so we use 'create_event'."
  ],
  "parameters": {
    "title": [
      "Interview with candidate",
      "Reflecting the user's request for an interview."
    ],
    "date": [
      "03-10",
      "Convert March 10th to '03-10'."
    ],
    "time": [
      "15:00",
      "3 PM => 15:00 in 24-hour format."
    ]
  }
}
"""

    combined = (
        "Below are some examples showing how to produce the JSON. "
        "Always produce only one JSON object with absolutely no extra text.\n\n"
        f"{needed_function_example}\n\n"
        f"{no_function_example}\n\n"
        f"{create_event_example}\n"
    )
    return combined


def build_system_prompt_with_functions(function_docs: Dict[str, Any]) -> str:
    domain_instructions = (
        "**You must classify the conversation domain among one of the following three**:\n\n"
        "1) Persuasion_Deliberation_and_Negotiation\n"
        "   - Focuses on resolving conflicts or reconciling differing viewpoints.\n"
        "   - Participants propose offers, trade-offs, or compromises.\n\n"
        "2) Inquiry_and_Information_Seeking\n"
        "   - Focuses on exploring unknowns or clarifying information.\n"
        "   - Participants ask detailed questions, verify facts, or confirm knowledge.\n\n"
        "3) Eristic\n"
        "   - Focuses on antagonism or hostility, aiming to 'win' an argument.\n"
        "   - Participants often use personal attacks or emotional appeals.\n\n"
        "Include your final domain guess in the JSON under 'domain': [<domain_name>, <short reason>].\n"
    )

    function_list_str = []
    for fn in function_docs["functions"]:
        fname = fn["function"]
        desc = fn["desc"]
        params = fn["parameters"]
        param_info = ", ".join([f"{p['name']}({p['type']})" for p in params])
        function_list_str.append(f"- {fname}: {desc}\n  Params: {param_info}")

    joined = "\n".join(function_list_str)
    examples_text = build_few_shot_example()

    system_prompt = (
        "You are a helpful assistant that can call the following functions when necessary. "
        "You MUST produce only a single valid JSON object in your final answer. "
        "Do NOT include any extra text, disclaimers, or formatting outside the JSON. "
        "If the user request or conversation implies a need to use one of these functions, do so. "
        "If no function is relevant, output a JSON with 'function': ['none', 'Reason'], 'parameters': {}.\n\n"
        "Domain Classification:\n"
        f"{domain_instructions}\n\n"
        "Tool Documentation:\n"
        f"{joined}\n\n"
        "Your final answer MUST be in this JSON format:\n"
        "{\n"
        '  "function": [\n'
        '    "<function_name or none>",\n'
        '    "A detailed explanation, in multiple sentences if needed, of why this function is selected or none."\n'
        "  ],\n"
        '  "parameters": {\n'
        '    "<param_name>": [\n'
        '      "<value>",\n'
        '      "A more detailed reason, possibly multiple sentences, explaining how this value was chosen."\n'
        "    ],\n"
        "    ...\n"
        "  },\n"
        '  "domain": [\n'
        '    "<domain_name>",\n'
        '    "Reasoning why the conversation is that domain."\n'
        "  ],\n"
        "}\n\n"
        "=== Guidelines for the second element of 'function' array ===\n"
        " - Summarize key conversation points that justify the function choice.\n"
        " - Clearly state the user's goal or request.\n"
        " - If alternative functions might apply, mention why they are not used.\n\n"
        "=== Guidelines for the second element in each parameter array ===\n"
        " - Provide a more detailed reasoning for why this particular value is used.\n"
        " - Refer to relevant parts of the conversation or user input.\n"
        " - Mention any constraints or preferences stated by the user.\n\n"
        "Here are some illustrative examples:\n\n"
        f"{examples_text}\n\n"
        "At the final step, produce exactly one JSON object with no extra text. End of instructions."
    )
    return system_prompt


def build_ai_response(parsed_json: Dict[str, Any], gold_label) -> str:
    prompt = f"""Here is the function: {parsed_json}
    and the virtual returned value of the function and parameter: {gold_label}

    You are an AI assistant, Given the function, parameter values, domain, and returned values, generate a AI Assistant-like response to the user.
    """
    return prompt


def map_speaker_to_role(speaker_name: str) -> str:
    lower_name = speaker_name.lower()
    if lower_name.startswith("agent_"):
        return "user"
    elif lower_name == "system":
        return "system"
    else:
        return "assistant"


def round_msgs_to_conversation(
    round_msgs: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    conv_list = []

    for msg in round_msgs:
        speaker = msg["speaker"]
        text = msg["message"]
        role = map_speaker_to_role(speaker)

        formatted_content = f"'{speaker}': {text}"
        conv_list.append({"role": role, "content": formatted_content})

    conv_list.pop()

    return conv_list


def round_msgs_to_unformatted_conversation(
    round_msgs: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    conv_list = []
    ret = ""
    for idx, msg in enumerate(round_msgs):
        if idx == len(round_msgs) - 1:
            break
        ret += f"{msg['speaker']}: {msg['message']}\n"

    conv_list.append({"role": "user", "content": ret})

    return conv_list


def extract_code_block(text: str) -> str:
    pattern = r"```(?:json)?([\s\S]*?)```"
    match = re.search(pattern, text, flags=re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return text.strip()


def extract_json_candidate(text: str) -> str:
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        return ""
    return text[first_brace : last_brace + 1]


def parse_json_with_reasoning(response_text: str) -> Dict[str, Any]:
    candidate = extract_json_candidate(response_text.strip())
    if not candidate:
        logger.error("No JSON candidate found in response text.")
        logger.error(f"Response text: {response_text}")
        return {}

    try:
        obj = json.loads(candidate)

        keys_to_extract = ["function", "parameters", "domain"]
        filtered_obj = {}
        for key in keys_to_extract:
            if key in obj:
                filtered_obj[key] = obj[key]

        return filtered_obj

    except json.JSONDecodeError:
        return {}
