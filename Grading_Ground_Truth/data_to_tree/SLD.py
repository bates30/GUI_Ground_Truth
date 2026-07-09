"""
Simple Leaf Distance Analysis
Direct hop distance comparison between trees

This script compares trees by calculating hop distances between all leaf pairs
and measuring how much each leaf's distances differ between trees.

"""

import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque, defaultdict
import numpy as np

# Paths
base_dir = Path(__file__).resolve().parent.parent
output_dir = Path(__file__).resolve().parent / "output"


def load_tree(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def get_all_leaves(tree_data):
    leaves = []
    for node in tree_data['nodes']:
        if node['kind'] == 'leaf':
            leaves.append(node['name'])
    return sorted(leaves)


def build_graph_from_json_with_root(tree_data, new_root):
    #Build a directed NetworkX graph from tree JSON data, rooted at new_root
    # Build adjacency map (undirected)
    adjacency = defaultdict(list)
    node_kinds = {}

    for node in tree_data['nodes']:
        node_kinds[node['name']] = node['kind']
        for child in node['children']:
            adjacency[node['name']].append(child)
            adjacency[child].append(node['name'])

    # BFS from new_root to build directed tree
    G = nx.DiGraph()
    G.add_node(new_root, kind=node_kinds[new_root])

    visited = {new_root}
    queue = deque([new_root])

    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                G.add_node(neighbor, kind=node_kinds[neighbor])
                G.add_edge(current, neighbor)
                queue.append(neighbor)

    return G


def hierarchical_layout(G, root, horizontal_spacing=1.8, vertical_spacing=2.5):
    #Create a top-down hierarchical layout for tree visualization
    if root not in G.nodes():
        return {}

    # BFS to determine levels and parent-child relationships
    levels = {root: 0}
    children = defaultdict(list)
    queue = deque([root])

    while queue:
        node = queue.popleft()
        for child in G.successors(node):
            levels[child] = levels[node] + 1
            children[node].append(child)
            queue.append(child)

    # Sort children for consistent layout
    for parent in children:
        children[parent] = sorted(children[parent])

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

    calc_widths(root)

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

    assign_positions(root, 0, 0)
    return pos


def create_leaf_distance_matrix(G, leaves):

    #Create a distance matrix between leaves based on tree path lengths (number of hops).
    n = len(leaves)
    distance_matrix = np.zeros((n, n))

    # Convert to undirected for shortest path calculation
    G_undirected = G.to_undirected()

    for i, leaf1 in enumerate(leaves):
        for j, leaf2 in enumerate(leaves):
            if i != j:
                try:
                    distance_matrix[i, j] = nx.shortest_path_length(G_undirected, leaf1, leaf2)
                except nx.NetworkXNoPath:
                    distance_matrix[i, j] = n * 2  # Large distance if no path

    return distance_matrix


def compute_distance_difference_scores(bible_leaves, bible_dist_matrix, test_leaves, test_dist_matrix, common_leaves):

    # Compute distance difference score for each leaf

    # For each leaf, calculate how much its distances to all other leaves differ between the two trees.
    # Higher score = leaf's position differs more structurally between trees.

    # Args:
    #     bible_leaves: List of leaves in bible tree
    #     bible_dist_matrix: Distance matrix for bible tree
    #     test_leaves: List of leaves in test tree
    #     test_dist_matrix: Distance matrix for test tree
    #     common_leaves: List of common leaves between both trees

    # Returns a Dictionary mapping each common leaf to its normalized difference score

    scores = {}

    # Create index mappings
    bible_leaf_to_idx = {leaf: i for i, leaf in enumerate(bible_leaves)}
    test_leaf_to_idx = {leaf: i for i, leaf in enumerate(test_leaves)}

    # For each common leaf, calculate total distance difference
    for leaf in common_leaves:
        bible_idx = bible_leaf_to_idx[leaf]
        test_idx = test_leaf_to_idx[leaf]

        total_diff = 0.0

        # Compare distances to all other common leaves
        for other_leaf in common_leaves:
            if other_leaf != leaf:
                other_bible_idx = bible_leaf_to_idx[other_leaf]
                other_test_idx = test_leaf_to_idx[other_leaf]

                # Get distance from this leaf to other leaf in both trees
                bible_distance = bible_dist_matrix[bible_idx, other_bible_idx]
                test_distance = test_dist_matrix[test_idx, other_test_idx]

                # Add absolute difference
                total_diff += abs(bible_distance - test_distance)

        scores[leaf] = total_diff

    # Normalize to 0-1 range
    max_score = max(scores.values()) if scores.values() else 1
    min_score = min(scores.values()) if scores.values() else 0

    if max_score > min_score:
        normalized = {leaf: (scores[leaf] - min_score) / (max_score - min_score)
                     for leaf in scores}
    else:
        normalized = {leaf: 0.0 for leaf in scores}

    return normalized


def get_node_color_from_score(score):
    #Map hop distance difference score to color
    # Color gradient: lightblue -> yellow -> orange -> red
    if score < 0.33:
        # Blue to yellow
        ratio = score / 0.33
        r = int(173 + (255 - 173) * ratio)
        g = int(216 + (255 - 216) * ratio)
        b = int(230 + (0 - 230) * ratio)
    elif score < 0.67:
        # Yellow to orange
        ratio = (score - 0.33) / 0.34
        r = 255
        g = int(255 - (255 - 165) * ratio)
        b = 0
    else:
        # Orange to red
        ratio = (score - 0.67) / 0.33
        r = 255
        g = int(165 - 165 * ratio)
        b = 0

    return f'#{r:02x}{g:02x}{b:02x}'


def visualize_trees_with_distance_coloring(bible_tree, test_tree, bible_root, test_root,
                                            distance_scores, common_leaves,
                                            bible_only_leaves, test_only_leaves,
                                            avg_distance_diff, output_path):
    #Visualize both trees side by side with hop distance difference coloring
    # Build directed graphs using their respective roots
    bible_G = build_graph_from_json_with_root(bible_tree, bible_root)
    test_G = build_graph_from_json_with_root(test_tree, test_root)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # Calculate layouts
    bible_pos = hierarchical_layout(bible_G, bible_root)
    test_pos = hierarchical_layout(test_G, test_root)

    # Assign colors based on distance scores - colors must match between trees for same leaves
    def get_node_colors_and_sizes(G, tree_root, only_leaves_this_tree):
        colors = []
        sizes = []
        for node in G.nodes():
            if node == tree_root:
                colors.append('green')
                sizes.append(2500)  # Larger
            elif G.nodes[node]['kind'] == 'leaf':
                if node in distance_scores:
                    # Use the SAME color for this leaf in both trees
                    colors.append(get_node_color_from_score(distance_scores[node]))
                    sizes.append(2000)  # Larger
                elif node in only_leaves_this_tree:
                    # Purple for Bible-only, dark green for Test-only
                    colors.append('purple' if only_leaves_this_tree == bible_only_leaves else 'darkgreen')
                    sizes.append(2200)  # Larger
                else:
                    colors.append('lightblue')
                    sizes.append(2000)  # Larger
            else:  # Junction, cluster, or source
                colors.append('lightgray')
                sizes.append(1400)  # Larger
        return colors, sizes

    bible_colors, bible_sizes = get_node_colors_and_sizes(bible_G, bible_root, bible_only_leaves)
    test_colors, test_sizes = get_node_colors_and_sizes(test_G, test_root, test_only_leaves)

    # Draw Bible tree 
    nx.draw_networkx_edges(bible_G, pos=bible_pos, ax=ax1, edge_color="gray",
                           width=4, arrows=False)  
    nx.draw_networkx_nodes(bible_G, pos=bible_pos, ax=ax1, node_color=bible_colors,
                          node_size=bible_sizes)
    nx.draw_networkx_labels(bible_G, pos=bible_pos, ax=ax1, font_color="black",
                           font_size=18, font_weight='bold') 

    ax1.set_title(f"Ground Truth (rooted at {bible_root})\n(Leafs colored by hop distance difference)",
                  fontsize=28, fontweight='bold')  
    ax1.axis('off')

    # Draw Test tree
    nx.draw_networkx_edges(test_G, pos=test_pos, ax=ax2, edge_color="gray",
                           width=4, arrows=False)  
    nx.draw_networkx_nodes(test_G, pos=test_pos, ax=ax2, node_color=test_colors,
                          node_size=test_sizes)
    nx.draw_networkx_labels(test_G, pos=test_pos, ax=ax2, font_color="black",
                           font_size=18, font_weight='bold') 

    ax2.set_title(f"Modeled Topology (rooted at {test_root})\n(Leafs colored by hop distance difference)",
                  fontsize=28, fontweight='bold') 
    ax2.axis('off')

    # Add legend 
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor='lightblue', label='Low hop distance difference'),
        Patch(facecolor='yellow', label='Medium hop distance difference'),
        Patch(facecolor='red', label='High hop distance difference'),
        Patch(facecolor='lightgray', label='Junction'),
        Patch(facecolor='green', label='Root nodes')
    ]

    # Add main title with metric
    fig.suptitle(f'Leaf Distance Comparison | Avg Distance Difference: {avg_distance_diff:.2f} hops',
                 fontsize=32, fontweight='bold', y=0.98)  

    fig.legend(handles=legend_items, loc='lower center', fontsize=16, ncol=3,
               bbox_to_anchor=(0.5, -0.01))  

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_metrics_json(avg_distance_diff, output_dir):
    
    #Save/update average leaf distance to Metrics.json in the output directory
    
    # Define metrics file path
    metrics_path = output_dir / "Metrics.json"

    # Read existing metrics if file exists
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {}

    # Update with average leaf distance
    metrics_data["avg_leaf_distance"] = avg_distance_diff

    # Write back to file
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)




def main(generate_image=True, GT_Tree = "Ground_Truth_CB1.json", Test_Tree = "CMST_CB1.json"):

    #Calculate average leaf distance and optionally generate visualization

    # Accept either full paths or filenames relative to base_dir
    new_tree_json = Path(GT_Tree) if Path(GT_Tree).is_absolute() else base_dir / "model_data" / GT_Tree
    consensus_tree_json = Path(Test_Tree) if Path(Test_Tree).is_absolute() else base_dir / "model_data" / Test_Tree
    bible_tree = load_tree(new_tree_json)
    test_tree = load_tree(consensus_tree_json)

    # Get leaves
    bible_leaves = get_all_leaves(bible_tree)
    test_leaves = get_all_leaves(test_tree)
    common_leaves = sorted(set(bible_leaves) & set(test_leaves))
    bible_only_leaves = sorted(set(bible_leaves) - set(test_leaves))
    test_only_leaves = sorted(set(test_leaves) - set(bible_leaves))

    # Use roots from JSON files
    bible_root = bible_tree['root']
    test_root = test_tree['root']

    # Build graphs for distance analysis
    bible_graph = build_graph_from_json_with_root(bible_tree, bible_root)
    test_graph = build_graph_from_json_with_root(test_tree, test_root)

    # Create distance matrices (number of hops between each pair of leaves)
    bible_dist = create_leaf_distance_matrix(bible_graph, bible_leaves)
    test_dist = create_leaf_distance_matrix(test_graph, test_leaves)

    # Calculate average distance difference (before normalization)
    leaf_to_bible = {leaf: i for i, leaf in enumerate(bible_leaves)}
    leaf_to_test = {leaf: i for i, leaf in enumerate(test_leaves)}

    total_diff = 0.0
    count = 0
    for leaf in common_leaves:
        for other in common_leaves:
            if leaf != other:
                bible_dist_val = bible_dist[leaf_to_bible[leaf], leaf_to_bible[other]]
                test_dist_val = test_dist[leaf_to_test[leaf], leaf_to_test[other]]
                total_diff += abs(bible_dist_val - test_dist_val)
                count += 1

    avg_distance_diff = total_diff / count if count > 0 else 0.0

    # Conditionally generate visualization
    if generate_image:
        # Compute distance difference scores for each leaf
        # This measures how much each leaf's distances to other leaves differ between trees
        distance_scores = compute_distance_difference_scores(bible_leaves, bible_dist, test_leaves, test_dist, common_leaves)

        # Visualize trees with distance difference coloring
        output_dir.mkdir(exist_ok=True)  # Create output directory if needed
        output_path = output_dir / "simple_leaf_distance_output.png"
        visualize_trees_with_distance_coloring(
            bible_tree, test_tree, bible_root, test_root, distance_scores,
            common_leaves, bible_only_leaves, test_only_leaves, avg_distance_diff, output_path
        )

    # Save average leaf distance metric to Metrics.json
    save_metrics_json(avg_distance_diff, output_dir)

