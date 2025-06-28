import json
import networkx as nx
import matplotlib.pyplot as plt
import click


@click.command()
@click.option(
    "--file_path",
    "-fp",
    default="tool_graph_v2.json",
    help="The directory where the graph_desc.json file is located.",
)
def visialize_graph(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()

    for node in data["nodes"]:
        G.add_node(node["function"])

    for link in data["links"]:
        G.add_edge(link["source"], link["target"])

    pos = nx.kamada_kawai_layout(G, scale=3)

    plt.figure(figsize=(60, 60), dpi=80)
    nx.draw_networkx_nodes(G, pos, node_color="#A4BBA6", node_size=1200)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=80)
    nx.draw_networkx_labels(
        G, pos, font_size=50, font_color="#78A5DD", font_weight="bold"
    )

    plt.axis("off")
    plt.tight_layout()

    output_image_path = file_path.replace(".json", ".pdf")
    plt.savefig(output_image_path, format="pdf", dpi=80)
    plt.close()


if __name__ == "__main__":
    visialize_graph()
