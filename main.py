import click
import yaml
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

@click.option('--yaml_path', default=None, help='predefined yaml import')
def main(agent, round, fewshot, iter, domain, functions, yaml_path):
    if yaml_path is not None:
        with open(yaml_path, encoding='utf-8') as f:
            yaml_data = yaml.full_load(f)
        agent = yaml_data['agent']
        round = yaml_data['round']
        fewshot = yaml_data['fewshot']
        iter = yaml_data['iter']
        domain = yaml_data['domain']
        functions = yaml_data['functions']

    # 1. Define few-shot prompt from path (TODO)
    fewshot = fewshot
    # 2. Define functions from path (TODO)
    funclist = BaseFunctionList(functions=functions)

    # 3. Define langgraph pipeline
    pm = PromptMaker(agent, round, fewshot, funclist, domain=domain)
    main_graph = make_agent_pipeline(pm)

    # 4. Get new data
    events_list = []
    for i in range(iter):
        events = main_graph.stream(
            {
                'messages': [
                    HumanMessage(content=pm.data_prompt())
                ]
            },
            {'recursion_limit': 100},
        )
        events_list.append(events)

    # 5. Get unique file name and save new data
    output_file = get_unique_filename("outputs/test.json")
    save_data(events_list, output_file)

if __name__ == '__main__':
    main()