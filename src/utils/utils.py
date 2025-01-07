
def save_data(events, save_path=None):
    savestr = ""
    for event in events:
        if isinstance(event, dict):
            # event는 노드 이름(예: "mediator", "agent_a")을 키로 가지는 딕셔너리
            for node_name, node_data in event.items():
                # node_data 내부에 messages 키가 있으면 메시지를 꺼낸다
                if "messages" in node_data:
                    messages = node_data["messages"]
                    # messages는 보통 [AIMessage(...), ...] 같은 리스트
                    for msg in messages:
                        # msg.content만 출력
                        print(f"[{node_name}] {msg.content}")
                        savestr += f"[{node_name}] {msg.content} \n"
        print("-" * 10)
        savestr += "-" * 10
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(savestr)