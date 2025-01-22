from src.validation.base import BaseValidator
from src.prompt.validation_prompt import validation_prompt_dict
import src.utils as utils
import json
from langchain_openai import ChatOpenAI


# GPT : get new func calls from dialogue, compare orig and new
class GraphValidator(BaseValidator):
    def __init__(self, tool_path):
        self.name = 'Graph'
        self.tool_doc = self.generate_tool_document(tool_path)
        self.valid_prompt = validation_prompt_dict[self.name].format(tool_document=self.tool_doc)
        self.client = ChatOpenAI(model='gpt-4o', temperature=0.2)
        self.base_messages = [
            (
                "system",
                self.valid_prompt
            )
        ]

    # Tool documentation 생성 함수
    def generate_tool_document(self, functions_path):
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


    def validate(self, data):
        # Do validation calculation
        # return average score for data.
        print("Validation for Graph Exact Match.")
        scores = []
        for i, data_i in enumerate(data):
            
            msg_i = data_i['messages']
            metadata_i = data_i['metadata']
            ref_i = metadata_i["functions"]
            messages = self.base_messages + [(
                "user",
                f"# DIALOGUE #: {msg_i}"
            )]
            result = self.client.invoke(messages)
            out_i = result.content
            try:
                if '`' in out_i:
                    out_i = out_i[7:-3]
                out_json = json.loads(out_i)
            except Exception as e:
                print(e)
                print(out_i)
            data_i['metadata']['val_functions'] = [fi['function_name'] for fi in out_json]
            data_i['metadata']['val_params_ret_val'] = [
                {
                    "function": fi['function_name'],
                    "parameters": {
                            p['parameter_name']: p['parameter_value']
                        for p in fi['parameters']
                    }
                }
                for fi in out_json
            ]
            if self.is_exact_match(data_i,i,len(data)):
                scores.append(1)
            else:
                scores.append(0)
            
        return data, scores


            
    def is_exact_match(self,data_i,i,l):
        print('-----------------------------')
        print(f'Data #{i+1}/{l}')
        func_match = data_i['metadata']['functions'] == data_i['metadata']['val_functions']
        if func_match:
            print("Func matched")
        else:
            print("Func unmatched")
        ref_data_val = data_i['metadata']['params_ret_val']
        out_data_val = data_i['metadata']['val_params_ret_val']
        for expected, actual in zip(ref_data_val, out_data_val):
            expected_params = expected['parameters']
            actual_params = actual['parameters']
            
            # Compare parameters
            try:
                all_match = all(
                    actual_params[param_name] == param_value[0]
                    for param_name, param_value in expected_params.items()
                )
            except Exception as e:
                print("Param name unmatched")
                print(e)
                print(actual_params)
                return 0
            if not all_match:
                print("Param value unmatched")
                print(e)
                print(actual_params)
                return 0
        print("Param all matched")
        return 1
