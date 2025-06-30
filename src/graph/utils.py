def is_sequential(G, node_list):
    for i in range(len(node_list) - 1):
        if not G.has_edge(node_list[i], node_list[i + 1]):
            return False

    return True
