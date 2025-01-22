import numpy as np
import json
from src.validation.graph import GraphValidator
from src.validation.ic import ICValidator
from src.validation.geval import GEvalValidator
from src.utils import utils

validator_dict = {
    "Graph": GraphValidator, "IC": ICValidator, "GEval": GEvalValidator
}

class ValidationSystem:
    def __init__(self, data, valid_types, data_path, tool_path):
        self.data = data
        self.result_path = '.'.join(data_path.split('.')[:-1]) + f"_{','.join([t for t in valid_types])}.json"
        self.validators = [validator_dict[valid_type](tool_path) for valid_type in valid_types]
    
    def get_result(self):
        score_dict = {}
        for validator in self.validators:
            data, scores = validator.validate(data = self.data)
            self.data = data
            print('-----------------------------')
            print(f'{validator.name} validation')
            print(f"Avg Result: {np.mean(scores)}")
            print('-----------------------------')
            score_dict[validator.name] = scores
        with open(self.result_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return score_dict