import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directories to path for imports
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "input_data"))

from input_data.Structure_for_Bus123 import get_positions
import input_data.Structure_for_Bus123 as Structure_for_Bus123

output_dir = Path(__file__).resolve().parent / "output"

changable_node_size = 750


def path_distance_from_graph(G, path):
    #Calculate total distance along a path in a graph
    return sum(float(G[u][v]["length"]) for u, v in zip(path[:-1], path[1:]))

B = nx.Graph()

# Global memory structures to persist across lists
yellow_node_connections = {}
all_pink_nodes = set()
all_yellow_nodes_history = set()
current_group_processed = set()


def add_nodes(bus_df, switch_df):
    G = nx.Graph()

    col1 = bus_df.iloc[:, 0].dropna().tolist()[1:]
    col2 = bus_df.iloc[:, 1].dropna().tolist()[1:]
    col3 = bus_df.iloc[:, 2].dropna().tolist()[1:]

    unique_nodes = set(col1).union(col2)
    G.add_nodes_from(unique_nodes)

    for u, v, length in zip(col1, col2, col3):
        G.add_edge(u, v, length=length)

    switch1 = switch_df.iloc[:, 0].dropna().tolist()[1:]
    switch2 = switch_df.iloc[:, 1].dropna().tolist()[1:]
    switch3 = switch_df.iloc[:, 2].dropna().tolist()[1:]

    for u, v, z in zip(switch1, switch2, switch3):
        try:
            if z == 'closed':
                G.add_edge(u, v, length=1)
        except:
            pass

    return G


def visualize_grid(red_node, yellow_nodes, source, G, pos, group_idx=0, output_path=None):
    global B, yellow_node_connections, all_pink_nodes, all_yellow_nodes_history, current_group_processed

    B.clear()
    B.add_node(red_node)

    fig, ax = plt.subplots(figsize=(16, 12))
    node_size = changable_node_size

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[source],
        node_size=node_size,
        node_shape="o",
        node_color="grey",
        ax=ax,
    )

    path = nx.shortest_path(G, source=red_node, target=source)
    between_nodes_probe = path

    probe_nodes = set(path[1:-1])
    path_nodes = set()

    yellow_nodes_first_instance = []
    red_nodes_first_instance = []
    new_intersection_nodes = []

    for y in yellow_nodes:
        try:
            path = nx.shortest_path(G, source=y, target=source)
            path_nodes.update(path[1:-1])

            intersection_node = None

            if path[0] in between_nodes_probe:
                intersection_node = next((n for n in path if n in between_nodes_probe), None)

                if intersection_node is not None:
                    nx.draw_networkx_nodes(
                        G,
                        pos,
                        nodelist=[intersection_node],
                        node_size=node_size * 1.5,
                        node_shape="o",
                        node_color="yellow",
                        ax=ax,
                    )
                    yellow_nodes_first_instance.append(y)

            elif y in between_nodes_probe:
                nx.draw_networkx_nodes(
                        G,
                        pos,
                        nodelist=[intersection_node],
                        node_size=node_size * 1.5,
                        node_shape="o",
                        node_color="yellow",
                )

            elif between_nodes_probe[0] in path:
                intersection_node = between_nodes_probe[0]
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=[intersection_node],
                    node_size=node_size * 1.5,
                    node_shape="o",
                    node_color="red",
                    ax=ax,
                )
                red_nodes_first_instance.append(intersection_node)

            else:
                intersection_node = next((n for n in path if n in between_nodes_probe), None)

            if intersection_node is not None:
                new_intersection_nodes.append(intersection_node)

        except nx.NetworkXNoPath:
            pass

    if yellow_nodes_first_instance:
        new_intersection_nodes += yellow_nodes_first_instance

    if red_nodes_first_instance:
        new_intersection_nodes += red_nodes_first_instance

    for i in all_pink_nodes:
        if i not in new_intersection_nodes:
            new_intersection_nodes.append(i)

    all_pink_nodes.update(new_intersection_nodes)

    previous_yellows_only = all_yellow_nodes_history - {red_node}

    for i in yellow_nodes:
        if i == red_node:
            continue
        try:
            yellow_path = nx.shortest_path(G, source=i, target=source)
            red_path = nx.shortest_path(G, source=red_node, target=source)

            best_node = None
            best_index = len(yellow_path)

            for idx, node in enumerate(yellow_path):
                if node in red_path and node in new_intersection_nodes and node != i:
                    if idx < best_index:
                        best_index = idx
                        best_node = node

            if best_node is None:
                for idx, node in enumerate(yellow_path):
                    if node in new_intersection_nodes and node != i:
                        if idx < best_index:
                            best_index = idx
                            best_node = node

            if best_node is not None:
                if i in yellow_node_connections:
                    old_connection = yellow_node_connections[i]
                    try:
                        old_idx = yellow_path.index(old_connection) if old_connection in yellow_path else len(yellow_path)

                        if best_index < old_idx:
                            yellow_node_connections[i] = best_node
                            B.add_edge(i, best_node)
                        else:
                            B.add_edge(i, old_connection)
                    except ValueError:
                        yellow_node_connections[i] = best_node
                        B.add_edge(i, best_node)
                else:
                    yellow_node_connections[i] = best_node
                    B.add_edge(i, best_node)

                Structure_for_Bus123.master_dict[i] = yellow_node_connections.get(i, best_node)

        except nx.NetworkXNoPath:
            pass

    for i in previous_yellows_only:
        if i in new_intersection_nodes:
            continue

        if i in yellow_node_connections:
            stored_connection = yellow_node_connections[i]
            B.add_edge(i, stored_connection)

    for i in new_intersection_nodes:
        try:
            path = nx.shortest_path(G, source=i, target=source)

            new_node = None
            for node in path[1:]:
                if node in new_intersection_nodes:
                    new_node = node
                    break

            if new_node is not None:
                B.add_edge(i, new_node)

        except nx.NetworkXNoPath:
            pass

    total_dist = float("inf")
    new_node = None
    for i in new_intersection_nodes:
        try:
            path = nx.shortest_path(G, source=i, target=source)
            new_dist = path_distance_from_graph(G, path)
            if new_dist < total_dist:
                total_dist = new_dist
                new_node = i
        except nx.NetworkXNoPath:
            pass

    if new_node is not None:
        B.add_edge(source, new_node)

    try:
        red_path = nx.shortest_path(G, source=red_node, target=source)

        closest_pink = None
        closest_index = len(red_path)

        for idx, node in enumerate(red_path):
            if node in new_intersection_nodes and node != red_node:
                if idx < closest_index:
                    closest_index = idx
                    closest_pink = node

        if closest_pink is not None:
            B.add_edge(red_node, closest_pink)

    except nx.NetworkXNoPath:
        B.add_edge(red_node, source)

    previous_yellow_nodes = all_yellow_nodes_history - {red_node}
    node_colors = []
    for n in B.nodes():
        if n == source:
            node_colors.append("green")
        elif n in new_intersection_nodes:
            node_colors.append("pink")
        elif n == red_node:
            node_colors.append("red")
        elif n in yellow_nodes:
            node_colors.append("yellow")
        elif n in previous_yellow_nodes:
            node_colors.append("orange")
        elif n in probe_nodes:
            node_colors.append("red")
        else:
            node_colors.append("yellow")

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        with_labels=False,
        node_color="lightgrey",
        font_color="black",
        node_size=50,
    )

    nx.draw_networkx_edges(
        B,
        pos=pos,
        ax=ax,
        edgelist=list(B.edges()),
        edge_color="red",
        width=2,
        style="solid",
    )

    if red_node in new_intersection_nodes:
        nx.draw_networkx_nodes(
            B,
            pos=pos,
            nodelist=[red_node],
            node_size=changable_node_size * 1.5,
            node_color="red",
            ax=ax,
        )

    for n in B.nodes():
        if n in new_intersection_nodes and n in yellow_nodes and n != red_node:
            nx.draw_networkx_nodes(
                B,
                pos=pos,
                ax=ax,
                nodelist=[n],
                node_size=changable_node_size * 1.5,
                node_color='yellow',
            )

    for n in B.nodes():
        if n in new_intersection_nodes and n in previous_yellow_nodes:
            nx.draw_networkx_nodes(
                B,
                pos=pos,
                ax=ax,
                nodelist=[n],
                node_size=changable_node_size * 1.5,
                node_color='orange',
            )

    nx.draw_networkx_nodes(
        B,
        pos=pos,
        ax=ax,
        nodelist=list(B.nodes()),
        node_color=node_colors,
        node_size=changable_node_size,
    )

    nx.draw_networkx_labels(
        B,
        pos=pos,
        ax=ax,
        font_color="black",
    )

    ax.invert_yaxis()
    ax.set_title(f"Probe location: {red_node} | Group: {group_idx}")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    results = {
        "new_intersection_nodes": list(new_intersection_nodes)
    }

    return results


def main(source_node, sensor_locations, switch_file_path, generate_image=True, probe_depth=1):

    #Generate grid visualization.

    # Args:
    #     source_node: Source/root node location for the tree (int)
    #     sensor_locations: List of listening sensor node locations (list of ints)
    #     switch_file_path: Path to the switch data Excel file (str or Path)
    #     generate_image: Whether to generate visualization image (bool)
    #     probe_depth: Number of sensors to use as probing sensors (int)

    # Returns:

    global yellow_node_connections, all_pink_nodes, all_yellow_nodes_history, current_group_processed

    # Reset global state
    yellow_node_connections.clear()
    all_pink_nodes.clear()
    all_yellow_nodes_history.clear()
    current_group_processed.clear()

    # Load data
    line_data_path = base_dir / "input_data" / "line_data.xls"
    bus_df = pd.read_excel(line_data_path)
    switch_df = pd.read_excel(switch_file_path)

    G = add_nodes(bus_df, switch_df)
    pos = get_positions()

    if not generate_image:
        return {}

    # Process sensor groups
    yellow_node_groups = [sensor_locations]
    results = []

    for group_idx, yellow_nodes in enumerate(yellow_node_groups):
        count = 0
        current_group_processed.clear()

        if probe_depth == 0:
            red_node_list = yellow_nodes
        else:
            red_node_list = yellow_nodes[:probe_depth]

        for idx, red_node in enumerate(red_node_list):
            count += 1
            other_yellow_nodes = [node for node in yellow_nodes if node != red_node]

            output_path = output_dir / f"Grid_Visual_group_{group_idx}_no_{count}_probe_{red_node}.png"

            result = visualize_grid(
                red_node,
                other_yellow_nodes,
                source_node,
                G,
                pos,
                group_idx=group_idx,
                output_path=output_path
            )
            results.append(result)

            current_group_processed.add(red_node)

        all_yellow_nodes_history.update(yellow_nodes)

    return results

