import json
import click
import atexit
import torch.distributed as dist
import re
from pathlib import Path

# Project utilities
from src.colorlog import get_logger
from src.config import settings

from transformers import AutoTokenizer

from vllm import SamplingParams


from . import utils
import torch._dynamo
import time


torch._dynamo.config.suppress_errors = True


@atexit.register
def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[0])
        dist.destroy_process_group()


logger = get_logger(__name__)

CUSTOM_CHAT_TEMPLATE = r"""
{% set bos = bos_token if bos_token else "" %}
{{ bos }} 
{% for msg in messages %}
{% if msg['role'] == 'system' %}
[SYSTEM] {{ msg['content'] }}

{% elif msg['role'] == 'user' %}
[USER] {{ msg['content'] }}

{% elif msg['role'] == 'assistant' %}
[ASSISTANT] {{ msg['content'] }}

{% else %}
[{{ msg['role']|upper() }}] {{ msg['content'] }}

{% endif %}
{% endfor %}

{% if add_generation_prompt %}
[ASSISTANT]
{% endif %}
"""


@click.command()
@click.option(
    "--model_name", default=settings.model_name, help="HF model path or repo."
)
@click.option(
    "--function_docs", default=settings.function_docs, help="Path to tool docs JSON."
)
@click.option(
    "--dataset_dir",
    default=settings.dataset_dir,
    help="Directory containing round_<n>.json files.",
)
@click.option(
    "--output_dir",
    default=settings.output_dir,
    help="Directory to save inference results.",
)
@click.option(
    "--max_tokens",
    default=settings.max_tokens,
    type=int,
    help="Maximum model context length.",
)
@click.option(
    "--tensor_parallel_size",
    default=settings.tensor_parallel_size,
    type=int,
    help="vLLM tensor parallel degree.",
)
def main(
    model_name, function_docs, dataset_dir, output_dir, max_tokens, tensor_parallel_size
):
    llm = utils.load_model(
        model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_tokens=max_tokens,
    )

    function_docs_path = function_docs
    function_docs_data = utils.load_function_docs(function_docs_path)

    for round_to_infer in range(1, 5):
        for round_data in range(round_to_infer, 5):
            formatted_name = re.sub(
                r"[^a-zA-Z0-9]", "_", model_name.split("/")[-1]
            ).lower()

            dialogue_path = f"{dataset_dir}/round_{round_data}.json"
            logger.info(
                f"🚀 Start inferencing for {dialogue_path} with model {model_name}"
            )
            output_file = Path(
                f"{output_dir}/{formatted_name}/{formatted_name}_round_{round_data}.json"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            data = utils.load_dialogue_data(dialogue_path)

            system_prompt = utils.build_system_prompt_with_functions(function_docs_data)

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            vllm_list = []

            round_name = f"Round {round_to_infer}"
            for diag in data:
                conversation = [
                    {"role": "system", "name": "system", "content": system_prompt}
                ]

                accumulated_messages = []
                for rdx, round in enumerate(diag["messages"]):
                    for round_name_inner, messages in round.items():
                        for msg in messages:
                            accumulated_messages.append(msg)
                    if rdx + 1 == round_to_infer:
                        break

                unformatted_conversation = (
                    conversation
                    + utils.round_msgs_to_unformatted_conversation(accumulated_messages)
                )
                formatted_conversation = (
                    conversation
                    + utils.round_msgs_to_conversation(accumulated_messages)
                )

                final_instruction = [
                    {
                        "role": "system",
                        "name": "system",
                        "content": (
                            "Consider the previous context, but focus particularly on the most recent round of conversation.\n"
                            "Given the tool documentation and the conversation, generate an appropriate function name, parameter values, and domain along with reasoning for their selection.\n"
                            "Do NOT add any extra text, disclaimers, or explanations.\n"
                            "If a function is appropriate, call it with reason.\n"
                            "Your final output must follow the specified JSON format strictly.\n"
                        ),
                    }
                ]
                unformatted_conversation += final_instruction
                formatted_conversation += final_instruction

                try:
                    prompt_text = tokenizer.apply_chat_template(
                        formatted_conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    conversation_for_list = formatted_conversation
                except Exception:
                    tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE
                    prompt_text = tokenizer.apply_chat_template(
                        unformatted_conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    conversation_for_list = unformatted_conversation

                vllm_list.append(prompt_text)

            logger.info(f"🚀 {model_name} has {len(vllm_list)} dialogues for inference")

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=2000,
            )

            start_time = time.time()
            outputs = llm.generate(vllm_list, sampling_params)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"🚀 Inference execution time: {elapsed_time:.2f} seconds")

            inference_results = []
            invalid_cnt, valid_cnt = 0, 0
            out_data = None
            if round_to_infer > 1 and output_file.exists():
                with open(output_file, "r") as f:
                    out_data = json.load(f)

            for idx, diag in enumerate(data):
                diag_id = diag["metadata"].get("diag_id", -1)

                diag_result = {"diag_id": diag_id, "round_results": []}

                model_output = outputs[idx].outputs[0].text
                try:
                    model_output = utils.extract_code_block(model_output)
                    _ = json.loads(model_output)
                    parsed_json = utils.parse_json_with_reasoning(model_output)
                    valid_cnt += 1
                except json.JSONDecodeError:
                    parsed_json = {
                        "function": "ERROR",
                        "parameters": {},
                        "reasoning": "JSON Decode Error",
                    }
                    invalid_cnt += 1

                if round_to_infer == 1:
                    round_result = {
                        "round_name": round_name,
                        "parsed_json": parsed_json,
                    }
                    diag_result["round_results"].append(round_result)

                else:
                    round_result = out_data[idx]["round_results"] if out_data else []
                    round_result.append(
                        {
                            "round_name": round_name,
                            "parsed_json": parsed_json,
                        }
                    )
                    diag_result["round_results"] = round_result

                inference_results.append(diag_result)

            logger.info(
                f"🎉 {model_name} has Valid JSONs: {valid_cnt}, Invalid JSONs: {invalid_cnt}"
            )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(inference_results, f, indent=4, ensure_ascii=False)

            logger.info(f"🎉 Inference results saved to {output_file}")


if __name__ == "__main__":
    main()
