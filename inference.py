import click
from vllm import LLM, SamplingParams
import json
import yaml
from src.prompt.inference import inference_prompt, tokenizer_template



# Tool documentation 생성 함수
def generate_tool_document(functions_path):
    tool_list = json.load(open(functions_path, "r"))["nodes"]
    for tool in tool_list:
        parameter_list = []
        for parameter in tool["parameters"]:
            parameter_list.append({
                "name": parameter["name"],
                "description": parameter["desc"]
            })
        tool["parameters"] = parameter_list

    document = "TOOL DOCUMENTATION\n"
    document += "Below is a list of tools with their descriptions and required parameters.\n"

    for tool in tool_list:
        document += f"TOOL ID: {tool['function']}\n"
        document += f"DESCRIPTION: {tool['desc']}\n"
        document += "PARAMETERS:\n"
        for param in tool['parameters']:
            document += f"  - {param['name']} ({param.get('type', 'string')}): {param['description']}\n"
        document += "\n"  # 각 도구를 구분하기 위한 빈 줄 추가

    return document

# transform to mpc data
def transform_to_mpc(datas):
    conversation_datas = []
    for _, data in datas.items():
        conversation_data = []
        for msg in data:
            if msg['name'] != "orchestrator":
                conversation_data.append({
                    "role": "user",
                    "name": msg['name'],
                    "content": msg['content']
                })
        conversation_datas.append(conversation_data)
    return conversation_datas

# make template dialogue
def apply_template(data, tokenizer):
    dialogue = tokenizer.apply_chat_template(
        data,
        tokenize=False,
        add_generation_prompt=True
    )
    return dialogue

# check output
def check_data(datas, tool_document, llm, tokenizer):
    sampling_params = SamplingParams(temperature=0.85, top_p=0.95, max_tokens=1024, repetition_penalty=1.1)
    data_outputs = []
    for data in datas:
        outputs = []
        for temp_dialogue in data:
            input_prompt = inference_prompt.replace('{{tool_document}}', tool_document).replace('{{dialogue}}', apply_template(temp_dialogue, tokenizer))

            output = llm.generate(input_prompt, sampling_params)
            outputs.append(output)
        data_outputs.append(outputs)
    return data_outputs



@click.command()
@click.option('--functions_path', default=None, help='func list')
@click.option('--data_path', default=None, help='data path, json')

@click.option('--yaml_path', default=None, help='predefined yaml import')
def main(functions_path, data_path, yaml_path):
    if yaml_path is not None:
        with open(yaml_path, encoding='utf-8') as f:
            yaml_data = yaml.full_load(f)
        functions_path = yaml_data['functions_path']
        data_path = yaml_data['data_path']
    tool_document = generate_tool_document(functions_path)
    with open(data_path, 'r') as file:
        datas = json.load(file)
    datas = transform_to_mpc(datas)
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=4096, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    tokenizer.chat_template = tokenizer_template
    outputs = check_data(datas, tool_document, llm, tokenizer)
    print(outputs[0])



if __name__ == '__main__':
    main()