import click
from langchain_core.messages import (
    HumanMessage
)
from src.agent.base import make_agent_pipeline
from src.prompt.base import PromptMaker
from src.function.base import BaseFunctionList
from src.utils.utils import save_data

@click.command()
@click.option('--agent', default=3, help='Number of agents.')
@click.option('--turn', default=1, help='Number of turns')
@click.option('--fewshot', default="", help='fewshot use')
def main(agent, turn, fewshot):
    # 1. Define few-shot prompt from path (TODO)
    fewshot = fewshot
    # 2. Define functions from path (TODO)
    funclist = BaseFunctionList()

    # 3. Define langgraph pipeline

    pm = PromptMaker(agent, fewshot, funclist)
    main_graph = make_agent_pipeline(pm)

    # 4. Get new data
    events = main_graph.stream(
        {
            'messages': [
                HumanMessage(content=pm.data_prompt())
            ]
        },
        {'recursion_limit': 100},
    )

    # 5. Show / Save new data
    save_data(events, "outputs/test.txt")

if __name__ == '__main__':
    main()