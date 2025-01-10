import click
import yaml
import json
from langchain_core.messages import (
    HumanMessage
)
from src.agent.base import make_agent_pipeline
from src.prompt.base import PromptMaker
from src.function.base import BaseFunctionList
from src.utils.utils import save_data
import os

def get_unique_filename(filename):
    """Check if a file exists and increment the filename with numbers if it does."""
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base}{counter}{ext}"
        counter += 1
    return filename

@click.command()
@click.option('--agent', default=3, help='Number of agents.')
@click.option('--round', default=1, help='Number of rounds')
@click.option('--fewshot', default="", help='fewshot use')
@click.option('--iter', default=1, help='number of iteration')
@click.option('--domain', default="Persuasion", help='domain')
@click.option('--functions', default=None, help='func list')
@click.option('--output_path', default="outputs/test.json", help='output_path')

@click.option('--yaml_path', default=None, help='predefined yaml import')
def main(agent, round, fewshot, iter, domain, functions, output_path, yaml_path):
    if yaml_path is not None:
        with open(yaml_path, encoding='utf-8') as f:
            yaml_data = yaml.full_load(f)
        agent = yaml_data['agent']
        round = yaml_data['round']
        fewshot = yaml_data['fewshot']
        iter = yaml_data['iter']
        domain = yaml_data['domain']
        functions = yaml_data['functions']
        output_path = yaml_data['output_path']

    # 1. Define few-shot prompt from path (TODO)
    fewshot = fewshot
    # 2. Define functions from path (TODO)
    funclist = BaseFunctionList(functions=functions)

    # 3. Define langgraph pipeline
    pm = PromptMaker(agent, round, fewshot, funclist, domain, iter)
    main_graph = make_agent_pipeline(pm)

    # 4. Get new data
    events_list = []
    domain_list = pm.now_domains
    for i in range(iter):
        data_prompt = pm.data_prompt()
        events = main_graph.stream(
            {
                'messages': [
                    HumanMessage(content=data_prompt)
                ]
            },
            {'recursion_limit': 100},
        )
        events_list.append(events)

    # 5. Get unique file name and save new data
    output_file = get_unique_filename(output_path)
    save_dicts, metadata_dicts = save_data(events_list, output_file)
    metadata_path = '.'.join(output_file.split('.')[:-1]) + '_metadata.json'
    metadata = {
        "domain": domain_list,
        "round": round,
        "funclist": funclist.func_doc,
        "orchestrator": metadata_dicts
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

if __name__ == '__main__':
    main()