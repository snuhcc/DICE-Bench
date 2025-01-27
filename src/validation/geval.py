from src.validation.base import BaseValidator
from src.prompt.validation_prompt import validation_prompt_dict
import src.utils as utils
import json
from langchain_openai import ChatOpenAI

class GEvalValidator(BaseValidator):
    def __init__(self, tool_path):
        self.name = 'GEval'
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
        print("Evaluation for GEVAL Method.")