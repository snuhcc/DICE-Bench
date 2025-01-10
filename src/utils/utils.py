import json
def save_data(events_list, save_path=None):
    save_dicts = {}
    metadata_dicts = {}
    for i, events in enumerate(events_list):
        print(f"#{i}th data\n")
        save_list = []
        metadata_list = []
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
                            if node_name != "orchestrator":
                                save_list.append({
                                    "name": node_name,
                                    "content": msg.content
                                })
                            else:
                                metadata_list.append({
                                    "content": msg.content
                                })
            print("-" * 10)

        save_dicts[f"data{i}"] = save_list
        metadata_dicts[f"data{i}"] = metadata_list
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(save_dicts, f)
        # json.dumps(save_dicts, indent=4)

    return save_dicts, metadata_dicts