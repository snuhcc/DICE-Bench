# Only libraries required by the remaining functionality
import random
import networkx as nx

from src.graph.utils import is_sequential


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

        self._directed_graph = nx.DiGraph()
        self._directed_graph.add_nodes_from(self.node_id_list)
        for lk in self.links:
            s = lk["source"]
            t = lk["target"]
            if s in self.node_id_list and t in self.node_id_list:
                self._directed_graph.add_edge(s, t)

    def sample_subgraph(self, subgraph_type="node", rounds_num=1):
        """Return a subgraph consisting of either:
        1. a single random node ("node")
        2. a sequential directed path of length ``rounds_num`` ("directed_graph")

        Only these two modes are required by the main pipeline. Any other
        ``subgraph_type`` value is treated as invalid.
        """
        if subgraph_type == "node":
            return self._sample_subgraph_node()
        elif subgraph_type == "directed_graph":
            return self._sample_subgraph_directed_graph(rounds_num)
        else:
            raise ValueError(f"Invalid subgraph_type: {subgraph_type}")

    def _sample_subgraph_node(self):
        chosen_fn = random.choice(self.node_id_list)
        G = nx.DiGraph()
        G.add_node(chosen_fn)
        return G

    def _sample_subgraph_directed_graph(self, rounds_num):
        if rounds_num < 1:
            raise ValueError("rounds_num should be >= 1")

        all_paths = []
        for start in self.node_id_list:
            stack = [(start, [start])]
            while stack:
                current, path = stack.pop()
                if len(path) == rounds_num:
                    all_paths.append(path)
                    continue
                successors = list(self._directed_graph.successors(current))
                for nxt in successors:
                    if nxt not in path:
                        stack.append((nxt, path + [nxt]))

        if not all_paths:
            print(f"No directed path of length {rounds_num} found.")
            return nx.DiGraph()

        chosen_path = random.choice(all_paths)
        subG = nx.DiGraph()
        subG.add_nodes_from(chosen_path)
        for i in range(len(chosen_path) - 1):
            subG.add_edge(chosen_path[i], chosen_path[i + 1])
        return subG

    def sample_node(self):
        G1 = self.sample_subgraph("node")
        return list(G1.nodes())

    def sample_graph(self, rounds_num):
        G3 = self.sample_subgraph("directed_graph", rounds_num=rounds_num)

        while not is_sequential(G3, list(G3.nodes())):
            print("Retrying... in sample_directed_graph")
            G3 = self.sample_subgraph("directed_graph", rounds_num=rounds_num)

        return [node for node in G3.nodes()]

# NOTE: The previous CLI / demo entry-point has been removed because it was not
# used anywhere in the library or pipeline. If graph sampling visualisation is
# required again in the future, consider creating a dedicated script under the
# "scripts/" directory.
