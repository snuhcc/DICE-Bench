import os
import json
import yaml
from pathlib import Path
import re

import click
from src.validation.system import ValidationSystem
from src.utils import utils

@click.command()
@click.option("--yaml_path", default=None, help="Path to a predefined YAML file.")
def main(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    data_path = yaml_data['data_path']
    tool_path = yaml_data['tool_path']
    valid_types = yaml_data['valid_types']
    valid_output_path = yaml_data['valid_output_path']
    # valid_output_path = utils.create_unique_output_path(valid_output_path, "valid_test")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vs = ValidationSystem(data, valid_types, data_path, tool_path)
    results_dict = vs.get_result()
    results_list = [
        {
            "index": i,
            "data": data[i],
            "validation_result": {
                valid_types[j]: results_dict[valid_types[j]][i]
                for j in range(len(valid_types))
            }
        }
        for i in range(len(data))
    ]
    # with open(valid_output_path, "w", encoding="utf-8") as f:
    #     json.dump(results_list, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()