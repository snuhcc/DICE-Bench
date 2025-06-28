import functools
import re
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI

from langchain_core.messages import ToolMessage, AIMessage, BaseMessage
from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def find_next_agent(message):
    if hasattr(message, "content"):
        text = message.content
    else:
        text = str(message)

    match = re.search(r"\[NEXT:\s*([^\]]+)\]", text)
    if match:
        next_agent = match.group(1)
        return next_agent
    return None


def router(state) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    next_agent = find_next_agent(last_message)
    try:
        if (
            "I'm sorry" in last_message.content
            and "I can't assist" in last_message.content
        ):
            return "__end__"
        if next_agent is None:
            return "continue"
        if "agent" in next_agent:
            return next_agent
        elif next_agent == "END" or "END" in last_message.content:
            return "__end__"
        return "continue"
    except Exception as e:
        print(next_agent)
        raise e


def create_agent(llm, prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=(prompt)),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt | llm


def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        result = AIMessage(content="ToolMessage processed", name=name)
    else:
        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
    }


def make_agent_pipeline(pm):
    agent_nodes = []
    cond_dict = {"__end__": END}
    llm = ChatOpenAI(model="gpt-4o", temperature=0.8)
    agent_names = [f"agent_{chr(97 + i)}" for i in range(pm.agent_num)]
    for i in range(pm.agent_num):
        ag = create_agent(
            llm,
            f"You are {agent_names[i]}. " + pm.agent_prompt(chr(97 + i), pm.agent_num),
        )
        agent_nodes.append(functools.partial(agent_node, agent=ag, name=agent_names[i]))
        cond_dict[agent_names[i]] = agent_names[i]
    ag = create_agent(llm, pm.agent_prompt("orch", pm.agent_num))
    orchestrator_node = functools.partial(agent_node, agent=ag, name="orchestrator")
    cond_dict["continue"] = "orchestrator"
    workflow = StateGraph(AgentState)
    for i, ag_node in enumerate(agent_nodes):
        workflow.add_node(agent_names[i], ag_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_conditional_edges("orchestrator", router, cond_dict)
    for cond_key in cond_dict.keys():
        if "agent" in cond_key:
            workflow.add_edge(cond_key, "orchestrator")
    workflow.add_edge(START, "orchestrator")
    graph = workflow.compile()
    return graph
