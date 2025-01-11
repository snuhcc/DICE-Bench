# utils.py

import networkx as nx

def check_tree_level_count(levels, expected_num_levels, nodes_per_level):
    """트리 레벨 개수 및 각 레벨의 노드 수 확인."""
    actual_num_levels = len(levels)
    if actual_num_levels != expected_num_levels:
        print(f"[경고] 레벨 개수가 예상({expected_num_levels})과 다릅니다: {actual_num_levels}")
        return False
    print(f"[확인] 레벨 개수: {actual_num_levels} (OK)")

    for i, lvl in enumerate(levels):
        if len(lvl) != nodes_per_level:
            print(f"[경고] 레벨 {i}의 노드 개수가 예상({nodes_per_level})과 다릅니다: {len(lvl)}")
            return False
        print(f"[확인] 레벨 {i} 노드 개수: {len(lvl)} (OK)")
    return True

def check_tree_connectivity(T, levels):
    """레벨 간 실제 간선 연결 상태 확인."""
    print()
    if len(levels) < 2:
        print("[확인] 레벨이 2개 미만이므로, 연결을 확인할 필요가 없습니다.")
        return []
    
    connection_list = []
    
    for i in range(len(levels) - 1):
        parent_nodes = levels[i]
        child_nodes = levels[i + 1]
        connections_map = {}

        for p in parent_nodes:
            connected_children = []
            for c in child_nodes:
                if T.has_node(p) and T.has_node(c) and T.has_edge(p, c):
                    connected_children.append(c)
            if connected_children:
                connections_map[p] = connected_children

        if not connections_map:
            print(f"[경고] 레벨 {i} → 레벨 {i+1} 사이에 연결이 전혀 없습니다.")
            return []
        else:
            print(f"[확인] 레벨 {i} → 레벨 {i+1} 연결 요약:")
            for parent, children_list in connections_map.items():
                connection_list.append((parent, children_list))
                print(f"  - {parent} → [{', '.join(children_list)}]")
    print()
    return connection_list

def print_tree_text(T, levels):
    """(T, levels)를 텍스트 형태로 트리 구조 출력."""
    visited = set()

    def print_subtree(node, level_idx, prefix, is_last):
        branch_symbol = "└─ " if is_last else "├─ "
        print(f"{prefix}{branch_symbol}{node}")

        if node in visited:
            print(f"{prefix}{'   ' if is_last else '│  '}(중복)")
            return
        visited.add(node)

        if level_idx >= len(levels) - 1:
            return

        next_level_nodes = levels[level_idx + 1]
        children = [c for c in next_level_nodes if T.has_edge(node, c)]

        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            sub_prefix = prefix + ("   " if is_last else "│  ")
            print_subtree(child, level_idx + 1, sub_prefix, child_is_last)

    if not levels or not levels[0]:
        print("(empty levels)")
        return

    roots = levels[0]
    for i, root in enumerate(roots):
        root_is_last = (i == len(roots) - 1)
        print_subtree(root, 0, prefix="", is_last=root_is_last)
        print()

def is_sequential(G, node_list):
    """G가 directed path 형태로 node_list가 순차 연결되어 있는지 확인."""
    for i in range(len(node_list) - 1):
        if not G.has_edge(node_list[i], node_list[i + 1]):
            print(f"Nodes {node_list[i]} → {node_list[i+1]} 가 순차적으로 연결되어 있지 않습니다.")
            return False
    print("Nodes are sequentially connected in the directed path.")
    return True