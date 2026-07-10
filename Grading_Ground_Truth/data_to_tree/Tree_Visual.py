import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import sys
from collections import defaultdict, deque

# Add parent directories to path for imports
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "input_data"))

import input_data.Structure_for_Bus123 as Structure_for_Bus123

output_dir = Path(__file__).resolve().parent / "output"

changable_node_size = 750

B = nx.DiGraph()

# Global memory structures
yellow_node_connections = {}
all_pink_nodes = set()
all_yellow_nodes_history = set()
current_group_processed = set()


def path_distance_from_graph(G, path):
    return sum(float(G[u][v]["length"]) for u, v in zip(path[:-1], path[1:]))


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
        if z == 'closed':
            G.add_edge(u, v, length=1)
        else:
            pass

    return G


def hierarchical_layout(tree_graph, source_node, horizontal_spacing=2.0, vertical_spacing=1.5, intersection_nodes=None):
    # Create a top-down hierarchical layout for tree visualization

    if source_node not in tree_graph.nodes():
        return {}

    if intersection_nodes is None:
        intersection_nodes = set()

    # BFS to determine levels and parent-child relationships
    levels = {source_node: 0}
    children = defaultdict(list)
    queue = deque([source_node])

    while queue:
        node = queue.popleft()
        for child in tree_graph.successors(node):
            levels[child] = levels[node] + 1
            children[node].append(child)
            queue.append(child)

    # Sort children: intersection nodes first (left), then leaf nodes (right)
    # This matches TED/RF/SLD where junction nodes branch left
    for parent in children:
        def sort_key(node):
            # Return (is_leaf, node_value)
            # is_leaf=False (0) for intersection nodes (they sort first - go left)
            # is_leaf=True (1) for leaf nodes (they sort last - go right)
            is_leaf = node not in intersection_nodes
            return (is_leaf, node)
        children[parent] = sorted(children[parent], key=sort_key)

    # Calculate subtree widths
    subtree_widths = {}
    def calc_widths(node):
        if node in subtree_widths:
            return subtree_widths[node]
        if not children[node]:
            subtree_widths[node] = 1
        else:
            subtree_widths[node] = sum(calc_widths(child) for child in children[node])
        return subtree_widths[node]

    calc_widths(source_node)

    # Assign positions
    pos = {}
    def assign_positions(node, x_start, level):
        y = -level * vertical_spacing

        if not children[node]:
            # Leaf node
            pos[node] = (x_start, y)
        else:
            # Internal node - center over children
            x_current = x_start
            for child in children[node]:
                assign_positions(child, x_current, level + 1)
                x_current += subtree_widths[child] * horizontal_spacing

            # Center parent over children
            child_positions = [pos[child][0] for child in children[node]]
            parent_x = (min(child_positions) + max(child_positions)) / 2
            pos[node] = (parent_x, y)

    assign_positions(source_node, 0, 0)
    return pos


def visualize_tree(red_node, yellow_nodes, source, G, group_idx=0, count=0, output_path=None):
    global B, yellow_node_connections, all_pink_nodes, all_yellow_nodes_history, current_group_processed

    B.clear()
    B.add_node(red_node)

    fig, ax = plt.subplots(figsize=(16, 12))

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
                    yellow_nodes_first_instance.append(y)

            elif y in between_nodes_probe:
                pass

            elif between_nodes_probe[0] in path:
                intersection_node = between_nodes_probe[0]
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
                            B.add_edge(best_node, i)
                        else:
                            B.add_edge(old_connection, i)
                    except ValueError:
                        yellow_node_connections[i] = best_node
                        B.add_edge(best_node, i)
                else:
                    yellow_node_connections[i] = best_node
                    B.add_edge(best_node, i)

                Structure_for_Bus123.master_dict[i] = yellow_node_connections.get(i, best_node)

        except nx.NetworkXNoPath:
            pass
    for i in previous_yellows_only:
        if i in new_intersection_nodes:
            continue

        if i in yellow_node_connections:
            stored_connection = yellow_node_connections[i]
            B.add_edge(stored_connection, i)

    for i in new_intersection_nodes:
        try:
            path = nx.shortest_path(G, source=i, target=source)

            new_node = None
            for node in path[1:]:
                if node in new_intersection_nodes:
                    new_node = node
                    break

            if new_node is not None:
                B.add_edge(new_node, i)

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
            B.add_edge(closest_pink, red_node)

    except nx.NetworkXNoPath:
        pass
    previous_yellow_nodes = all_yellow_nodes_history - set(yellow_nodes) - {red_node}

    nodes_will_split = []
    for n in B.nodes():
        if n in new_intersection_nodes:
            if (n in yellow_nodes and n != red_node) or (n in previous_yellow_nodes):
                nodes_will_split.append(n)

    h_spacing = 5.0 if nodes_will_split else 2.5
    tree_pos = hierarchical_layout(B, source, horizontal_spacing=h_spacing, vertical_spacing=3.0, intersection_nodes=new_intersection_nodes)

    disconnected_nodes = [n for n in B.nodes() if n not in tree_pos]
    if disconnected_nodes:
        if tree_pos:
            min_x = min(pos[0] for pos in tree_pos.values())
            max_y = max(pos[1] for pos in tree_pos.values())
        else:
            min_x = 0
            max_y = 0

        for idx, node in enumerate(disconnected_nodes):
            tree_pos[node] = (min_x - 5.0, max_y - idx * 3.0)

    nodes_to_split = []

    for n in B.nodes():
        if n in new_intersection_nodes:
            if n in yellow_nodes and n != red_node:
                nodes_to_split.append((n, 'yellow'))
            elif n in previous_yellow_nodes:
                nodes_to_split.append((n, 'orange'))

    split_node_mapping = {}

    def is_position_clear(pos_x, pos_y, existing_positions, min_distance=2.5):
        for ex_x, ex_y in existing_positions:
            distance = ((pos_x - ex_x)**2 + (pos_y - ex_y)**2)**0.5
            if distance < min_distance:
                return False
        return True

    for original_node, color_type in nodes_to_split:
        yellow_orange_node = f"{original_node}_listening"

        B.add_node(yellow_orange_node)

        if original_node in tree_pos:
            orig_x, orig_y = tree_pos[original_node]

            existing_positions = [pos for node, pos in tree_pos.items() if node != original_node]

            attempts = [
                (-4.0, 0.0), (-4.0, 1.0), (-4.0, -1.0), (-4.0, 2.0), (-4.0, -2.0),
                (-5.0, 0.0), (-5.0, 1.0), (-5.0, -1.0), (-6.0, 0.0), (-6.0, 1.5),
                (-6.0, -1.5), (-4.0, 3.0), (-4.0, -3.0), (-7.0, 0.0), (-8.0, 0.0),
            ]

            new_pos = None
            for offset_x, offset_y in attempts:
                test_x = orig_x + offset_x
                test_y = orig_y + offset_y
                if is_position_clear(test_x, test_y, existing_positions):
                    new_pos = (test_x, test_y)
                    break

            if new_pos is None:
                new_pos = (orig_x - 8.0, orig_y)

            tree_pos[yellow_orange_node] = new_pos

        B.add_edge(yellow_orange_node, original_node)

        split_node_mapping[original_node] = (original_node, yellow_orange_node, color_type)

    node_colors = []
    node_color_map = {}

    for n in B.nodes():
        color = None

        if isinstance(n, str) and "_listening" in str(n):
            original_node = int(str(n).replace("_listening", ""))
            if original_node in split_node_mapping:
                _, _, color_type = split_node_mapping[original_node]
                color = color_type
        elif n == source:
            color = "green"
        elif n in new_intersection_nodes:
            color = "pink"
        elif n == red_node:
            color = "red"
        elif n in previous_yellow_nodes:
            color = "orange"
        elif n in probe_nodes:
            color = "red"
        else:
            color = "yellow"

        node_colors.append(color)
        node_color_map[n] = color

    edges_to_draw_solid = list(B.edges())

    if edges_to_draw_solid:
        nx.draw_networkx_edges(
            B,
            pos=tree_pos,
            ax=ax,
            edgelist=edges_to_draw_solid,
            edge_color="red",
            width=2,
            style="solid",
        )

    if red_node in new_intersection_nodes and red_node not in [n for n, _ in nodes_to_split]:
        nx.draw_networkx_nodes(
            B,
            pos=tree_pos,
            nodelist=[red_node],
            node_size=changable_node_size * 1.5,
            node_color="red",
            ax=ax,
        )

    for n in B.nodes():
        if n in new_intersection_nodes and n in yellow_nodes and n != red_node:
            if n not in [node for node, _ in nodes_to_split]:
                nx.draw_networkx_nodes(
                    B,
                    pos=tree_pos,
                    ax=ax,
                    nodelist=[n],
                    node_size=changable_node_size * 1.5,
                    node_color='yellow',
                )

    for n in B.nodes():
        if n in new_intersection_nodes and n in previous_yellow_nodes:
            if n not in [node for node, _ in nodes_to_split]:
                nx.draw_networkx_nodes(
                    B,
                    pos=tree_pos,
                    ax=ax,
                    nodelist=[n],
                    node_size=changable_node_size * 1.5,
                    node_color='orange',
                )

    nx.draw_networkx_nodes(
        B,
        pos=tree_pos,
        ax=ax,
        nodelist=list(B.nodes()),
        node_color=node_colors,
        node_size=changable_node_size,
    )

    labels = {}
    for n in B.nodes():
        if isinstance(n, str) and "_listening" in str(n):
            original_node = int(str(n).replace("_listening", ""))
            labels[n] = str(original_node)
        else:
            labels[n] = str(n)

    nx.draw_networkx_labels(
        B,
        pos=tree_pos,
        labels=labels,
        ax=ax,
        font_color="black",
    )

    legend_items = [
        Patch(facecolor="green", label="Source"),
        Patch(facecolor="yellow", label="Listening Sensor (Current)"),
        #Patch(facecolor="orange", label="Listening Sensor (Previous)"),
        Patch(facecolor="red", label="Probing Sensor Location"),
        Patch(facecolor="pink", label="First Shared Node"),
        Line2D([0], [0], color='red', linewidth=2, linestyle='-', label='Connections (Solid)'),
    ]
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(-0.1, 1.1))

    ax.set_title(f"Probe location: {red_node}")
    ax.axis('off')

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    results = {
        "new_intersection_nodes": list(new_intersection_nodes)
    }

    return results


def main(source_node, sensor_locations, switch_file_path, generate_image=True, probe_depth=1):
    ""
    # Generate tree visualization

    # Returns:
    #     dict: Results from visualization

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

            output_path = output_dir / f"Tree_Visual_group_{group_idx}_no_{count}_probe_{red_node}.png"

            result = visualize_tree(
                red_node,
                other_yellow_nodes,
                source_node,
                G,
                group_idx=group_idx,
                count=count,
                output_path=output_path
            )
            results.append(result)

            current_group_processed.add(red_node)

        all_yellow_nodes_history.update(yellow_nodes)

    return results
