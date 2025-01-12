import os
import json
import random
import networkx as nx
from matplotlib import pyplot as plt

from src.graph.utils import (
    check_tree_level_count,
    check_tree_connectivity,
    print_tree_text,
    is_sequential
)

class ToolGraphSampler:
    def __init__(self, tool_graph_json):
        self.tool_graph = tool_graph_json
        self.nodes = tool_graph_json["nodes"]
        self.links = tool_graph_json["links"]

        self.node_id_list = []
        self.node_dict = {}
        for nd in self.nodes:
            fn = nd["function"]
            self.node_id_list.append(fn)
            self.node_dict[fn] = nd

        # 방향 그래프 생성
        self._directed_graph = nx.DiGraph()
        self._directed_graph.add_nodes_from(self.node_id_list)
        for lk in self.links:
            s = lk["source"]
            t = lk["target"]
            if s in self.node_id_list and t in self.node_id_list:
                self._directed_graph.add_edge(s, t)

    def sample_subgraph(self, subgraph_type="node", num_nodes=3, num_levels=2):
        if subgraph_type == "node":
            return self._sample_subgraph_node()
        elif subgraph_type == "undirected_path":
            return self._sample_subgraph_undirected_path(num_nodes)
        elif subgraph_type == "directed_path":
            return self._sample_subgraph_directed_path(num_nodes)
        elif subgraph_type == "tree":
            return self._sample_subgraph_tree(num_levels=num_levels, nodes_per_level=num_nodes)
        else:
            raise ValueError(f"Invalid subgraph_type: {subgraph_type}")

    def _sample_subgraph_node(self):
        chosen_fn = random.choice(self.node_id_list)
        G = nx.DiGraph()
        G.add_node(chosen_fn)
        return G

    def _sample_subgraph_undirected_path(self, num_nodes):
        num_nodes = min(num_nodes, len(self.node_id_list))
        chosen_fns = random.sample(self.node_id_list, num_nodes)
        G = nx.Graph()
        G.add_nodes_from(chosen_fns)
        for i in range(num_nodes - 1):
            G.add_edge(chosen_fns[i], chosen_fns[i + 1])
        return G

    def _sample_subgraph_directed_path(self, num_nodes):
        if num_nodes < 1:
            raise ValueError("num_nodes should be >= 1")

        all_paths = []
        for start in self.node_id_list:
            stack = [(start, [start])]
            while stack:
                current, path = stack.pop()
                if len(path) == num_nodes:
                    all_paths.append(path)
                    continue
                successors = list(self._directed_graph.successors(current))
                for nxt in successors:
                    if nxt not in path:
                        stack.append((nxt, path + [nxt]))

        if not all_paths:
            print(f"No directed path of length {num_nodes} found.")
            return nx.DiGraph()

        chosen_path = random.choice(all_paths)
        subG = nx.DiGraph()
        subG.add_nodes_from(chosen_path)
        for i in range(len(chosen_path) - 1):
            subG.add_edge(chosen_path[i], chosen_path[i + 1])
        return subG

    def _sample_subgraph_tree(self, num_levels=2, nodes_per_level=2):
        """간단한 트리 구조의 서브그래프를 샘플링해 (DiGraph, levels)를 반환."""
        unused = set(self.node_id_list)
        T = nx.DiGraph()
        levels = []

        # 레벨 0 생성
        level0 = self._choose_level_with_at_least_one_outdegree(unused, nodes_per_level)
        for nd in level0:
            unused.remove(nd)
        T.add_nodes_from(level0)
        levels.append(level0)

        # 레벨 1 ~ num_levels-1
        for i in range(1, num_levels):
            prev_level = levels[i - 1]
            candidates = [p for p in prev_level if len(list(self._directed_graph.successors(p))) > 0]

            # 자식 있는 노드가 없으면 랜덤 배치
            if not candidates:
                next_level = self._choose_random_level(unused, nodes_per_level)
                for nd in next_level:
                    unused.remove(nd)
                T.add_nodes_from(next_level)
                levels.append(next_level)
                continue

            # 자식 노드가 있는 bridging_node 선정
            bridging_node = random.choice(candidates)
            successors = list(self._directed_graph.successors(bridging_node))
            unused_succ = [s for s in successors if s in unused]

            # 자식 중 아직 쓰이지 않은 노드가 없으면 랜덤 배치
            if not unused_succ:
                next_level = self._choose_random_level(unused, nodes_per_level)
                for nd in next_level:
                    unused.remove(nd)
                T.add_nodes_from(next_level)
                levels.append(next_level)
                continue

            # 자식 중 하나 고르고, 나머지 자리는 랜덤으로 채움
            chosen_child = random.choice(unused_succ)
            next_level = [chosen_child]
            unused.remove(chosen_child)

            remain_count = nodes_per_level - 1
            if remain_count > 0:
                remain_count = min(remain_count, len(unused))
                random_others = random.sample(list(unused), remain_count)
                for nd in random_others:
                    unused.remove(nd)
                next_level.extend(random_others)

            T.add_nodes_from(next_level)
            levels.append(next_level)
            for child_node in next_level:
                T.add_edge(bridging_node, child_node)

        return T, levels

    def _choose_level_with_at_least_one_outdegree(self, unused_nodes, k):
        outdegree_nodes = [n for n in unused_nodes if self._directed_graph.out_degree(n) > 0]
        if not outdegree_nodes:
            k = min(k, len(unused_nodes))
            return random.sample(list(unused_nodes), k)

        must_node = random.choice(outdegree_nodes)
        chosen = [must_node]
        tmp_unused = list(unused_nodes - {must_node})
        remain = k - 1
        remain = min(remain, len(tmp_unused))
        chosen += random.sample(tmp_unused, remain)
        return chosen

    def _choose_random_level(self, unused_nodes, k):
        k = min(k, len(unused_nodes))
        return random.sample(list(unused_nodes), k)

    @staticmethod
    def plot_subgraph(G, title="Subgraph", save_path=None):
        plt.figure(figsize=(7, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=800)
        nx.draw_networkx_labels(G, pos, font_size=9)

        if G.is_directed():
            nx.draw_networkx_edges(
                G, pos,
                arrows=True, arrowstyle='->', arrowsize=20,
                connectionstyle='arc3,rad=0.1', edge_color='blue'
            )
        else:
            nx.draw_networkx_edges(G, pos, edge_color='red')

        plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Graph saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    
    def sample_node(self):
        G1 = self.sample_subgraph("node")
        return list(G1.nodes())
    
    def sample_undirected_path(self, num_nodes=2):
        G2 = self.sample_subgraph("undirected_path", num_nodes=num_nodes)
        return list(G2.nodes())
    
    def sample_directed_path(self, num_nodes=2):
        G3 = self.sample_subgraph("directed_path", num_nodes=num_nodes)
        
        while not is_sequential(G3, list(G3.nodes())):
            print("Retrying... in sample_directed_path")
            G3 = self.sample_subgraph("directed_path", num_nodes=num_nodes)
            
        return list(G3.nodes())
    
    def sample_tree(self, num_levels=2, nodes_per_level=2):
        G4, levels4 = self.sample_subgraph("tree", num_nodes=nodes_per_level, num_levels=num_levels)
        
        level_count_ok = check_tree_level_count(levels4, num_levels, nodes_per_level)
        connection_list = check_tree_connectivity(self._directed_graph, levels4)

        while not (level_count_ok and len(connection_list)):
            print("Retrying...\n")
            G4, levels4 = self.sample_subgraph("tree", num_nodes=nodes_per_level, num_levels=num_levels)

            level_count_ok = check_tree_level_count(levels4, num_levels, nodes_per_level)
            connection_list = check_tree_connectivity(self._directed_graph, levels4)
        
        return levels4, connection_list 

def main():
    # 1) tool_graph.json 로드
    with open('tool_graph.json', 'r') as f:
        tool_graph = json.load(f)

    sampler = ToolGraphSampler(tool_graph)

    # 2) 그래프 저장 폴더 설정
    base_folder_name = "graph_plots/graph_sample_plot"
    folder_name = base_folder_name
    suffix_count = 0
    while os.path.exists(folder_name):
        suffix_count += 1
        folder_name = f"{base_folder_name}_{suffix_count}"
    os.makedirs(folder_name)
    print(f"Created folder: {folder_name}")

    # 1) node
    print("# 1) node")
    G1 = sampler.sample_subgraph("node")
    print("Nodes:", list(G1.nodes()))
    print("Edges:", list(G1.edges()))
    sampler.plot_subgraph(
        G1,
        title="Sample Node Graph",
        save_path=os.path.join(folder_name, "sample_node.png")
    )
    print()

    # 2) undirected_path
    print("# 2) undirected_path")
    G2 = sampler.sample_subgraph("undirected_path", num_nodes=3)
    print("Nodes:", list(G2.nodes()))
    print("Edges:", list(G2.edges()))
    sampler.plot_subgraph(
        G2,
        title="Sample Undirected Path Graph",
        save_path=os.path.join(folder_name, "sample_undirected_path.png")
    )
    print()

    # 3) directed_path
    print("# 3) directed_path")
    G3 = sampler.sample_subgraph("directed_path", num_nodes=4)
    
    while not is_sequential(G3, list(G3.nodes())):
        print("Retrying...")
        G3 = sampler.sample_subgraph("directed_path", num_nodes=4)
        
    print("Nodes:", list(G3.nodes()))
    print("Edges:", list(G3.edges()))
        
    sampler.plot_subgraph(
        G3,
        title="Sample Directed Path Graph",
        save_path=os.path.join(folder_name, "sample_directed_path.png")
    )
    print()

    # 4) tree
    print("# 4) tree")
    num_levels = 3
    nodes_per_level = 2
    G4, levels4 = sampler.sample_subgraph("tree", num_nodes=nodes_per_level, num_levels=num_levels)

    # 레벨 수, 노드 개수 체크
    level_count_ok = check_tree_level_count(levels4, num_levels, nodes_per_level)
    connection_list = check_tree_connectivity(sampler._directed_graph, levels4)

    while not (level_count_ok and len(connection_list)):
        print("Retrying...\n")
        G4, levels4 = sampler.sample_subgraph("tree", num_nodes=nodes_per_level, num_levels=num_levels)

        level_count_ok = check_tree_level_count(levels4, num_levels, nodes_per_level)
        connection_list = check_tree_connectivity(sampler._directed_graph, levels4)


    print("Nodes:", list(G4.nodes()))
    print("Edges:", list(G4.edges()))
    print("levels4:", levels4)
    print("connection_list:", connection_list)

    sampler.plot_subgraph(
        G4,
        title="Sample Tree Graph",
        save_path=os.path.join(folder_name, "sample_tree.png")
    )

    print("\n[텍스트 기반 '트리 구조' 출력]")
    print_tree_text(G4, levels4)
    print("Done!")


if __name__ == "__main__":
    main()