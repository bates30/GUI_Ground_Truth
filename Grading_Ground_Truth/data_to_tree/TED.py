"""

Using Zhang-Shasha Algorithm

"""

from zss import Node, distance
from typing import List
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque, defaultdict

# Paths
base_dir = Path(__file__).resolve().parent.parent

output_dir = Path(__file__).resolve().parent / "output"


def load_json_tree(filepath: str) -> dict:

    with open(filepath, 'r') as f:
        return json.load(f)


def json_tree_to_dict(tree_json: dict, root_name: str):

    #Convert JSON tree format to nested dictionary format for zss.

    #Returns Nested dictionary with 'label' and 'children' keys

    # Build a map of parent->children relationships
    node_map = {}
    for node in tree_json['nodes']:
        node_name = node['name']
        node_map[node_name] = node

    def build_recursive(node_name: str, visited: set):
        #Recursively build tree structure
        if node_name in visited:
            return None
        visited.add(node_name)

        node = node_map.get(node_name)
        if not node:
            return None

        # Determine label (internal nodes = '*', leaves = their name)
        # Handle all internal node types: junction, cluster, source
        if node['kind'] == 'leaf':
            label = node['name']
        else:
            label = '*'

        # Get children
        children = []
        if 'children' in node and node['children']:
            for child_name in node['children']:
                child_dict = build_recursive(child_name, visited)
                if child_dict:
                    children.append(child_dict)

        # Sort children to normalize ordering: leaves first (sorted by label), then internal nodes (by their children structure)
        # This ensures child order doesn't affect TED calculation
        children.sort(key=lambda c: (0 if c['label'] != '*' else 1, str(c['label']), str(sorted([ch['label'] for ch in c.get('children', [])]))))

        return {'label': label, 'children': children}

    return build_recursive(root_name, set())


def build_tree_from_dict(tree_dict: dict):

    #Build a zss Node tree from a dictionary structure.

    #Returns zss.Node object
    if tree_dict is None:
        return None

    label = tree_dict.get('label', '*')
    children = []

    if 'children' in tree_dict and tree_dict['children']:
        for child_dict in tree_dict['children']:
            child_node = build_tree_from_dict(child_dict)
            if child_node is not None:
                children.append(child_node)

    return Node(label, children)


def get_all_leaves(node: Node, leaves: List[str] = None):

    #Get all leaf labels from a tree.

    #Args:
    #    node: Root node of the tree
    #    leaves: Accumulator list (used in recursion)

    #Returns List of leaf labels

    if leaves is None:
        leaves = []

    if not node.children:
        # This is a leaf
        leaves.append(node.label)
    else:
        # Recurse on children
        for child in node.children:
            get_all_leaves(child, leaves)

    return leaves


def count_nodes(node: Node):

    #Count total nodes in tree.


    if node is None:
        return 0

    count = 1  # This node
    for child in node.children:
        count += count_nodes(child)

    return count


def get_subtree_leaves(node: Node):
    #Get all leaves under a given node
    leaves = []
    if not node.children:
        leaves.append(node.label)
    else:
        for child in node.children:
            leaves.extend(get_subtree_leaves(child))
    return leaves

#For coloring
def levenshtein_distance(seq1: List, seq2: List):

    # Calculate Levenshtein edit distance between two sequences

    # Returns Minimum edit distance

    len1, len2 = len(seq1), len(seq2)

    # Create DP table
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Match
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Delete from seq1
                    dp[i][j-1],      # Insert into seq1
                    dp[i-1][j-1]     # Replace
                )

    return dp[len1][len2]


def get_path_context(node: Node, target_leaf: str, path: List[Node] = None):

    # Get the structural path context from root to a target leaf.
    # Each element in the path includes only the sibling context (branching structure).
    # Internal node labels are ignored to ensure only tree structure matters.


    #Returns List of paths (as tuples of sibling_tuple only) for each occurrence of target_leaf

    if path is None:
        path = []

    # Create context for current node
    sibling_labels = tuple(sorted([child.label for child in node.children])) if node.children else ()

    # For internal nodes: only store branching structure (sibling_labels)
    # For leaf nodes: store the label itself
    # This ensures internal node names have NO effect on the comparison
    if node.children:
        # Internal node - only track the branching pattern, not the node's own label
        current_context = sibling_labels
    else:
        # Leaf node - store as tuple containing just the label for consistent typing
        current_context = (node.label,)

    new_path = path + [current_context]

    if not node.children:
        # This is a leaf
        if node.label == target_leaf:
            return [new_path]
        return []

    # Recurse on children
    all_paths = []
    for child in node.children:
        all_paths.extend(get_path_context(child, target_leaf, new_path))

    return all_paths


def calculate_subtree_distances(tree1: Node, tree2: Node):

    # Calculate TED for each leaf by measuring structural position differences
    # Uses path edit distance to compare the structural path from root to each leaf


    # Returns Dictionary mapping leaf labels to their structural edit contribution scores

    leaves1 = get_all_leaves(tree1)
    leaves2 = get_all_leaves(tree2)

    unique_to_tree1 = set(leaves1) - set(leaves2)
    unique_to_tree2 = set(leaves2) - set(leaves1)
    common_leaves = set(leaves1) & set(leaves2)

    # Pre-count occurrences for efficiency
    from collections import Counter
    count1_map = Counter(leaves1)
    count2_map = Counter(leaves2)

    node_scores = {}

    # Leaves unique to tree1 need DELETE - high score
    for leaf in unique_to_tree1:
        node_scores[leaf] = 3.0

    # Leaves unique to tree2 need INSERT - high score
    for leaf in unique_to_tree2:
        node_scores[leaf] = 3.0

    # For common leaves, calculate structural contribution using path edit distance
    for leaf in common_leaves:
        # Count occurrences in each tree
        count1 = count1_map[leaf]
        count2 = count2_map[leaf]

        if count1 != count2:
            # Different frequencies = structural edits needed
            node_scores[leaf] = min(abs(count1 - count2) * 2.0, 3.0)
        else:
            # Same frequency - compare structural paths using edit distance
            paths1 = get_path_context(tree1, leaf)
            paths2 = get_path_context(tree2, leaf)

            if paths1 and len(paths1) > 0 and paths2 and len(paths2) > 0:
                # For leaves appearing multiple times, compare all pairwise path distances
                # and use the minimum (best alignment)
                min_path_dist = float('inf')

                for p1 in paths1:
                    for p2 in paths2:
                        # Calculate edit distance between path contexts
                        path_dist = levenshtein_distance(p1, p2)
                        min_path_dist = min(min_path_dist, path_dist)

                # Path edit distance directly becomes the score
                # No cap - let it scale naturally for better differentiation
                node_scores[leaf] = float(min_path_dist)
            else:
                # Fallback if paths can't be found
                node_scores[leaf] = 0.5

    return node_scores


def analyze_node_involvement(tree1: Node, tree2: Node):

    # Analyze which nodes are involved in edit operations using structural analysis

    # Returns Dictionary mapping node labels to their involvement scores

    leaves1 = get_all_leaves(tree1)
    leaves2 = get_all_leaves(tree2)

    # Find unique leaves
    unique_to_tree1 = set(leaves1) - set(leaves2)
    unique_to_tree2 = set(leaves2) - set(leaves1)
    common_leaves = set(leaves1) & set(leaves2)

    # Pre-count occurrences for efficiency
    from collections import Counter
    count1_map = Counter(leaves1)
    count2_map = Counter(leaves2)

    # Get structural scores
    scores = calculate_subtree_distances(tree1, tree2)

    node_info = {}

    # Leaves unique to tree1 need DELETE
    for leaf in unique_to_tree1:
        node_info[leaf] = {
            'operation': 'delete',
            'score': scores[leaf],
            'type': 'leaf',
            'description': 'Unique to tree1 (needs DELETE)'
        }

    # Leaves unique to tree2 need INSERT
    for leaf in unique_to_tree2:
        node_info[leaf] = {
            'operation': 'insert',
            'score': scores[leaf],
            'type': 'leaf',
            'description': 'Unique to tree2 (needs INSERT)'
        }

    # Common leaves - score by structural position
    for leaf in common_leaves:
        count1 = count1_map[leaf]
        count2 = count2_map[leaf]
        score = scores[leaf]

        if count1 != count2:
            node_info[leaf] = {
                'operation': 'structural',
                'score': score,
                'type': 'leaf',
                'description': f'Appears {count1} times in tree1, {count2} times in tree2'
            }
        elif score > 1.0:
            node_info[leaf] = {
                'operation': 'structural',
                'score': score,
                'type': 'leaf',
                'description': f'Different structural path (path edit distance = {int(score)})'
            }
        else:
            node_info[leaf] = {
                'operation': 'match',
                'score': score,
                'type': 'leaf',
                'description': f'Similar structural path (path edit distance = {int(score)})'
            }

    return node_info


def is_leaf(node: Node):
    #Check if a node is a leaf (has no children)
    return not node.children


def custom_update_cost(node1: Node, node2: Node):

    # Cost of updating/changing a node label
    # Returns 0 for internal nodes (ignoring label changes) or when labels match
    # Returns 1 for leaf nodes with different labels

    # If both are internal nodes, label changes are free
    if not is_leaf(node1) and not is_leaf(node2):
        return 0

    # For leaf nodes or mixed cases, standard comparison
    return 0 if node1.label == node2.label else 1


def custom_insert_cost(node: Node):
    #Cost of inserting a node
    return 1


def custom_remove_cost(node: Node):
    #Cost of removing a node. Standard cost of 1
    return 1


def calculate_ted_with_path(tree1_dict: dict, tree2_dict: dict, count_internal_ops: bool = True):

    # Calculate Tree Edit Distance and analyze edit operations.


    # Returns Dictionary containing:
#         distance: Total edit distance
#         operations: List of inferred operations
#         node_scores: Scores for each node
#         leaf_analysis: Detailed leaf-by-leaf analysis

    # Build zss Node trees
    tree1 = build_tree_from_dict(tree1_dict)
    tree2 = build_tree_from_dict(tree2_dict)

    # Always use the cost functions that ignore internal node labels
    # and make internal node structure changes free unless count_internal_ops is True
    if count_internal_ops:
        # Count internal node structure changes
        def internal_aware_insert(node):
            return 1  # All insertions cost 1

        def internal_aware_remove(node):
            return 1  # All removals cost 1

        ted = distance(
            tree1,
            tree2,
            get_children=Node.get_children,
            insert_cost=internal_aware_insert,
            remove_cost=internal_aware_remove,
            update_cost=custom_update_cost  # 0 for internal, 0/1 for matching/different leaves
        )
    else:
        # Make internal node insert/remove operations free
        def zero_cost_internal_insert(node):
            return 0 if not is_leaf(node) else 1

        def zero_cost_internal_remove(node):
            return 0 if not is_leaf(node) else 1

        ted = distance(
            tree1,
            tree2,
            get_children=Node.get_children,
            insert_cost=zero_cost_internal_insert,
            remove_cost=zero_cost_internal_remove,
            update_cost=custom_update_cost
        )

    # Count nodes
    tree1_node_count = count_nodes(tree1)
    tree2_node_count = count_nodes(tree2)

    # Get leaf sets
    leaves1 = get_all_leaves(tree1)
    leaves2 = get_all_leaves(tree2)

    unique_to_tree1 = set(leaves1) - set(leaves2)
    unique_to_tree2 = set(leaves2) - set(leaves1)
    common_leaves = set(leaves1) & set(leaves2)

    # Calculate the number of internal nodes in each tree
    # Internal nodes = total nodes - leaf nodes
    tree1_internal_nodes = tree1_node_count - len(leaves1)
    tree2_internal_nodes = tree2_node_count - len(leaves2)

    # Analyze node involvement - this calculates path edit distances for each leaf
    node_info = analyze_node_involvement(tree1, tree2)

    # Extract node scores
    node_scores = {}
    for node_label, info in node_info.items():
        node_scores[node_label] = info['score']

    # Custom TED metric:
    # Count leaves that need to be inserted/deleted (different leaf sets)
    # Count leaves that moved depth (any depth change = 1 penalty, regardless of distance)
    # Count internal nodes that need to be inserted/removed

    # Leaf-only operations: insert/delete leaves that differ between trees
    leaf_edits = len(unique_to_tree1) + len(unique_to_tree2)

    # Calculate depth differences for all common leaves
    def get_leaf_depth(tree_node, target_leaf, depth=0):
        #Get the depth of a specific leaf in the tree
        if not tree_node.children:
            if tree_node.label == target_leaf:
                return depth
            return None
        for child in tree_node.children:
            result = get_leaf_depth(child, target_leaf, depth + 1)
            if result is not None:
                return result
        return None

    depth_diffs = {}
    total_depth_diff = 0
    moved_leaves = 0
    for leaf in common_leaves:
        depth1 = get_leaf_depth(tree1, leaf)
        depth2 = get_leaf_depth(tree2, leaf)
        if depth1 is not None and depth2 is not None:
            diff = abs(depth1 - depth2)
            depth_diffs[leaf] = (depth1, depth2, diff)
            total_depth_diff += diff
            if diff > 0:
                moved_leaves += 1

    # Structural edits: count of leaves that moved (1 penalty each, regardless of distance)
    leaf_depth_changes = moved_leaves

    # Internal node difference: each extra/missing internal node = 1 penalty
    internal_node_diff = abs(tree1_internal_nodes - tree2_internal_nodes)

    # Total adjusted TED: leaf inserts/deletes + leaves moved + internal node changes
    adjusted_ted = leaf_edits + leaf_depth_changes + internal_node_diff

    # Structural edits = leaves moved + internal nodes changed
    structural_edits = leaf_depth_changes + internal_node_diff

    # Build operations list
    operations = []

    for leaf in sorted(unique_to_tree1):
        operations.append({
            'type': 'delete',
            'node': leaf,
            'cost': 1,
            'description': f'Delete leaf {leaf} (only in tree1)',
            'comment': f'Leaf "{leaf}" exists in Ground Truth but not in Modeled Topology. Remove this node to match target tree.'
        })

    for leaf in sorted(unique_to_tree2):
        operations.append({
            'type': 'insert',
            'node': leaf,
            'cost': 1,
            'description': f'Insert leaf {leaf} (only in tree2)',
            'comment': f'Leaf "{leaf}" exists in Modeled Topology but not in Ground Truth. Add this node to match target tree.'
        })

    # Add structural operations (approximated)
    if structural_edits > 0:
        # Get leaves with structural changes for better commenting
        structural_leaves = [leaf for leaf in common_leaves
                           if node_info.get(leaf, {}).get('operation') == 'structural']

        operations.append({
            'type': 'structural',
            'node': 'multiple',
            'cost': structural_edits,
            'description': f'{structural_edits} structural edit(s) to reorganize common leaves',
            'comment': f'Reorganize the tree structure for {len(structural_leaves)} common leaves that appear in different positions or frequencies. '
                      f'This includes moving nodes to different parents or changing the hierarchical arrangement. '
                      f'Affected leaves: {", ".join(sorted(structural_leaves)[:5])}{"..." if len(structural_leaves) > 5 else ""}'
        })

    result = {
        'distance': int(adjusted_ted),
        'leaf_edits': leaf_edits,
        'structural_edits': int(structural_edits),
        'internal_node_diff': internal_node_diff,
        'moved_leaves': moved_leaves,
        'leaf_depth_changes': leaf_depth_changes,
        'operations': operations,
        'node_scores': node_scores,
        'node_info': node_info,
        'original_ted': int(ted),  # Keep the original TED for debugging
        'leaf_analysis': {
            'tree1_unique': sorted(list(unique_to_tree1)),
            'tree2_unique': sorted(list(unique_to_tree2)),
            'common': sorted(list(common_leaves)),
            'tree1_total': len(leaves1),
            'tree2_total': len(leaves2)
        },
        'tree_sizes': {
            'tree1_nodes': tree1_node_count,
            'tree2_nodes': tree2_node_count,
            'tree1_internal': tree1_internal_nodes,
            'tree2_internal': tree2_internal_nodes
        }
    }

    return result


def get_node_color_from_score(score, max_score):

    # Map edit distance score to color gradient.

    # Args:
    #     score: Node's edit distance contribution score
    #     max_score: Maximum score for normalization

    # Returns Hex color string

    if max_score <= 0:
        return '#add8e6'  # lightblue for no edits

    # Normalize score to 0-1 range
    normalized = min(score / max_score, 1.0)

    # Color gradient: lightblue, yellow, orange, red
    if normalized < 0.33:
        # Blue to yellow
        ratio = normalized / 0.33
        r = int(173 + (255 - 173) * ratio)
        g = int(216 + (255 - 216) * ratio)
        b = int(230 + (0 - 230) * ratio)
    elif normalized < 0.67:
        # Yellow to orange
        ratio = (normalized - 0.33) / 0.34
        r = 255
        g = int(255 - (255 - 165) * ratio)
        b = 0
    else:
        # Orange to red
        ratio = (normalized - 0.67) / 0.33
        r = 255
        g = int(165 - 165 * ratio)
        b = 0

    return f'#{r:02x}{g:02x}{b:02x}'


def build_graph_from_json_with_root(tree_data, root_name):
    #Build a directed NetworkX graph from tree JSON data.

    # Build adjacency map (undirected)
    adjacency = defaultdict(list)
    node_kinds = {}

    for node in tree_data['nodes']:
        node_kinds[node['name']] = node['kind']
        for child in node['children']:
            adjacency[node['name']].append(child)
            adjacency[child].append(node['name'])

    # BFS from root to build directed tree
    G = nx.DiGraph()
    G.add_node(root_name, kind=node_kinds[root_name])

    visited = {root_name}
    queue = deque([root_name])

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

    # Create a top-down hierarchical layout for tree visualization.


    # Returns a Dictionary mapping node names to (x, y) positions

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


def visualize_trees_with_ted_coloring(tree1_json, tree2_json, tree1_root, tree2_root,
                                     node_scores, tree1_unique, tree2_unique,
                                     tree1_name, tree2_name, ted_distance, max_ted, output_path):

    # Visualize both trees side by side with TED score coloring.

    # Args:
    #     tree1_json: First tree JSON data
    #     tree2_json: Second tree JSON data
    #     tree1_root: Root node name for tree1
    #     tree2_root: Root node name for tree2
    #     node_scores: Dictionary mapping node labels to TED scores
    #     tree1_unique: Set of leaves unique to tree1
    #     tree2_unique: Set of leaves unique to tree2
    #     tree1_name: Display name for tree1
    #     tree2_name: Display name for tree2
    #     ted_distance: The tree edit distance
    #     max_ted: Maximum possible tree edit distance
    #     output_path: Path to save the visualization

    # Build directed graphs
    tree1_G = build_graph_from_json_with_root(tree1_json, tree1_root)
    tree2_G = build_graph_from_json_with_root(tree2_json, tree2_root)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # Calculate layouts
    tree1_pos = hierarchical_layout(tree1_G, tree1_root)
    tree2_pos = hierarchical_layout(tree2_G, tree2_root)

    # Find max score for normalization
    max_score = max(node_scores.values()) if node_scores else 1.0


    # Assign colors and sizes (larger nodes)
    def get_node_colors_and_sizes(G, tree_root, unique_leaves_this_tree, unique_leaves_other_tree):
        colors = []
        sizes = []
        for node in G.nodes():
            if node == tree_root:
                colors.append('green')
                sizes.append(2500)  # Increased
            elif G.nodes[node]['kind'] == 'leaf':
                if node in unique_leaves_this_tree:
                    # Purple for tree1-only, dark green for tree2-only
                    colors.append('purple' if unique_leaves_this_tree == tree1_unique else 'darkgreen')
                    sizes.append(2200)  # Increased
                elif node in node_scores:
                    # Use same color for common leaves based on TED score
                    color = get_node_color_from_score(node_scores[node], max_score)
                    colors.append(color)
                    sizes.append(2000)  # Increased
                else:
                    colors.append('lightblue')
                    sizes.append(2000)  # Increased
            else:  # Junction, cluster, or source
                colors.append('lightgray')
                sizes.append(1400)  # Increased



        return colors, sizes

    tree1_colors, tree1_sizes = get_node_colors_and_sizes(tree1_G, tree1_root, tree1_unique, tree2_unique)
    tree2_colors, tree2_sizes = get_node_colors_and_sizes(tree2_G, tree2_root, tree2_unique, tree1_unique)

    # Draw Tree 1 (2x larger fonts and edges)
    nx.draw_networkx_edges(tree1_G, pos=tree1_pos, ax=ax1, edge_color="gray",
                           width=4, arrows=False)  # 2x from 2
    nx.draw_networkx_nodes(tree1_G, pos=tree1_pos, ax=ax1, node_color=tree1_colors,
                          node_size=tree1_sizes)
    nx.draw_networkx_labels(tree1_G, pos=tree1_pos, ax=ax1, font_color="black",
                           font_size=18, font_weight='bold')  # 2x from 9

    ax1.set_title(f"{tree1_name} (rooted at {tree1_root})\n(Leaves colored by edit distance contribution)",
                  fontsize=28, fontweight='bold')  # 2x from 14
    ax1.axis('off')

    # Draw Tree 2 (2x larger fonts and edges)
    nx.draw_networkx_edges(tree2_G, pos=tree2_pos, ax=ax2, edge_color="gray",
                           width=4, arrows=False)  # 2x from 2
    nx.draw_networkx_nodes(tree2_G, pos=tree2_pos, ax=ax2, node_color=tree2_colors,
                          node_size=tree2_sizes)
    nx.draw_networkx_labels(tree2_G, pos=tree2_pos, ax=ax2, font_color="black",
                           font_size=18, font_weight='bold')  # 2x from 9

    ax2.set_title(f"{tree2_name} (rooted at {tree2_root})\n(Leaves colored by edit distance contribution)",
                  fontsize=28, fontweight='bold')  # 2x from 14
    ax2.axis('off')

    # Add main title with metrics (2x larger font)
    fig.suptitle(f'Tree Edit Distance: {ted_distance} operations | Max TED: {max_ted}',
                 fontsize=32, fontweight='bold', y=0.98)  # 2x from 16

    # Add legend with score range (smaller font to avoid overlap)
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor='lightblue', label=f'Low path edit distance (score = 0)'),
        Patch(facecolor='yellow', label=f'Medium path edit distance'),
        Patch(facecolor='red', label=f'High path edit distance (score = {max_score:.1f})'),
        Patch(facecolor='lightgray', label='Junction'),
        Patch(facecolor='green', label='Root nodes')
    ]
    fig.legend(handles=legend_items, loc='lower center', fontsize=16, ncol=3,
               bbox_to_anchor=(0.5, -0.01))  # Reduced from 24 to 16, adjusted position

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_metrics_json(ted_distance, max_ted, output_dir):

    # Calculate similarity and save/update Metrics.json in the output directory.
    # Similarity = 1 - (TED distance / Max TED)

    # Calculate similarity
    if max_ted > 0:
        similarity = 1 - (ted_distance / max_ted)
    else:
        similarity = 1.0  # If max_ted is 0, trees are identical

    # Define metrics file path
    metrics_path = output_dir / "Metrics.json"

    # Read existing metrics if file exists
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {}

    # Update with TED similarity
    metrics_data["TED_similarity"] = similarity

    # Write back to file
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)



def main(generate_image=True, GT_Tree = "Ground_Truth_CB1.json", Test_Tree =  "CMST_CB1.json"):

    #Calculate TED similarity and optionally generate visualization.

    # Accept either full paths or filenames relative to base_dir
    #Ground_truth
    tree1_json_path = Path(GT_Tree) if Path(GT_Tree).is_absolute() else base_dir / "model_data" / GT_Tree


    #Test tree
    tree2_json_path = Path(Test_Tree) if Path(Test_Tree).is_absolute() else base_dir / "model_data" / Test_Tree
    tree1_name = "GT"
    tree2_name = "Model"

    tree1_json = load_json_tree(tree1_json_path)
    tree2_json = load_json_tree(tree2_json_path)

    tree1_root = tree1_json['root']
    tree2_root = tree2_json['root']

    tree1_dict = json_tree_to_dict(tree1_json, tree1_root)
    tree2_dict = json_tree_to_dict(tree2_json, tree2_root)

    # Set count_internal_ops=False to make internal node operations free (cost 0)
    # This ensures only leaf differences and structural changes affect the TED score
    result = calculate_ted_with_path(tree1_dict, tree2_dict, count_internal_ops=False)

    # Calculate max possible TED and similarity percentage
    max_ted = result['tree_sizes']['tree1_nodes'] + result['tree_sizes']['tree2_nodes']

    # Conditionally generate visualization
    if generate_image:
        output_path = output_dir / "TED_output.png"
        tree1_unique = set(result['leaf_analysis']['tree1_unique'])
        tree2_unique = set(result['leaf_analysis']['tree2_unique'])

        visualize_trees_with_ted_coloring(
            tree1_json, tree2_json, tree1_root, tree2_root,
            result['node_scores'], tree1_unique, tree2_unique,
            tree1_name, tree2_name, result['distance'], max_ted, output_path
        )

    # Save TED similarity metric to Metrics.json
    save_metrics_json(result['distance'], max_ted, output_dir)

