import functools
import re
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from typing import Literal

from langchain_core.messages import (
    ToolMessage,
    AIMessage,
    BaseMessage
)
from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def find_next_agent(message):
    # 메시지가 AIMessage 객체라면 content 속성을 사용
    if hasattr(message, "content"):
        text = message.content
    else:
        text = str(message)  # 그렇지 않으면 문자열로 변환 (예외 상황 대비)

    # 정규식을 사용하여 [NEXT: ...] 부분 추출
    match = re.search(r"\[NEXT: ([^\]]+)\]", text)
    
    if match:
        next_agent = match.group(1)  # 그룹 1에 원하는 값이 들어있음
        return next_agent
    return None  # 매칭되지 않을 경우 None 반환

def router(state) -> str:
    messages = state['messages']
    last_message = messages[-1]
    # print(state)
    next_agent = find_next_agent(last_message)  # `find_next_agent`는 다음 에이전트를 추출하는 함수
    try:
        if next_agent is None:
            return 'continue'
        if 'agent' in next_agent:
            return next_agent
        elif next_agent == 'END':
            return '__end__'  # 종료 상태
        return 'continue' 
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
    result = agent.invoke(state)  # LLM 호출
    
    if isinstance(result, ToolMessage):
        # ToolMessage에 대한 추가 처리
        result = AIMessage(content="ToolMessage processed", name=name)
    else:
        result = AIMessage(**result.model_dump(exclude={'type', 'name'}), name=name)

    
    return {
        'messages': [result],
    }

def make_agent_pipeline(pm):
    agent_nodes = []
    cond_dict = {
        '__end__': END
        }
    llm = ChatOpenAI(model='gpt-4o')
    agent_names = [f'agent_{chr(97+i)}' for i in range(pm.agent_num)]
    for i in range(pm.agent_num):
        ag = create_agent(llm, pm.agent_prompt(chr(97+i)))
        agent_nodes.append(functools.partial(agent_node, agent=ag, name=agent_names[i]))
        cond_dict[agent_names[i]] = agent_names[i]
    # orchestrator
    ag = create_agent(llm, pm.agent_prompt('orch'))
    orchestrator_node = functools.partial(agent_node, agent=ag, name="orchestrator")
    cond_dict['continue'] = 'orchestrator'
    # StateGraph 설정
    workflow = StateGraph(AgentState)

    # 상태 그래프 설정
    for i, ag_node in enumerate(agent_nodes):
        workflow.add_node(agent_names[i], ag_node)
    workflow.add_node("orchestrator", orchestrator_node)

    workflow.add_conditional_edges(
        'orchestrator',
        router,
        cond_dict
    )
    for cond_key in cond_dict.keys():
        if "agent" in cond_key:
            workflow.add_edge(cond_key, "orchestrator")
    workflow.add_edge(START, 'orchestrator')
    graph = workflow.compile()
    return graph

