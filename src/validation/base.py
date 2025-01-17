import numpy as np
from src.validation.graph import GraphValidator
from src.validation.ic import ICValidator
from src.validation.geval import GEvalValidator

validator_dict = {
    "Graph": GraphValidator, "IC": ICValidator, "GEval": GEvalValidator
}

class ValidationSystem:
    def __init__(self, data, reference, valid_types):
        self.data = data
        self.reference = reference
        self.validators = [validator_dict[valid_type] for valid_type in valid_types]
    
    def get_result(self):
        score_dict = {}
        for validator in self.validators:
            scores = validator.validate(self.data, self.reference)
            print('-----------------------------')
            print(f'{validator.name} validation')
            print(f"Avg Result: {np.mean(scores)}")
            print('-----------------------------')
            score_dict[validator.name] = scores
        return score_dict

class BaseValidator:
    def __init__(self):
        self.name = 'base'

    def validate(self, data, reference):
        # Do validation calculation
        # return score for each data. (list)
        pass

