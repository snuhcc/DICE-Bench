import json
import networkx as nx
import matplotlib.pyplot as plt
import click

@click.command()
@click.option('--data_dir', default='.', help='The directory where the graph_desc.json file is located.')
def visialize_graph(data_dir):
    graph_file = f"{data_dir}/tool_graph.json"
    with open(graph_file, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()

    for node in data["nodes"]:
        G.add_node(node["function"])

    for link in data["links"]:
        G.add_edge(link["source"], link["target"])

    # 레이아웃 계산
    pos = nx.kamada_kawai_layout(G)

    # 그래프 그리기
    plt.figure(figsize=(60, 60), dpi=80)
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=1200)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=40)
    nx.draw_networkx_labels(G, pos, font_size=50, font_color="green", font_weight="bold")

    # 축 제거 및 레이아웃 정리
    plt.axis("off")
    plt.tight_layout()

    # 이미지 파일로 저장
    output_image_path = graph_file.replace(".json", ".png")
    plt.savefig(output_image_path, format="png", dpi=80)
    plt.close()

if __name__ == "__main__":
    visialize_graph()