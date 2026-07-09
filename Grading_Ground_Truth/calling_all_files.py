"""
Orchestrator script that coordinates tree generation and comparison metrics.

This script:
1. Takes sensor locations and CB state as input
2. Calls new_leaf_structure_tree_v2.py to generate ground truth tree
3. Calls GUI_ready_RF.py, GUI_ready_SLD.py, GUI_ready_TED.py with generated and test trees
4. Outputs all metrics to Metrics.json
"""

import sys
from pathlib import Path

# Set up directory paths
# __file__ is in: gui_keep/Grading_Ground_Truth/calling_all_files.py
grading_dir = Path(__file__).resolve().parent  # Grading_Ground_Truth
parent_dir = grading_dir.parent  # gui_keep

# Add parent directory to sys.path to enable package imports
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from Grading_Ground_Truth.data_to_tree.new_leaf_structure import run_tree_generation


def main(source_node, sensor_locations, CB_state, test_tree_path,
         output_dir=None, probe_depth=1):


    # Step 1: Extract CB state number from filename (e.g., "CB1.xls" -> 1)
    if isinstance(CB_state, str):
        if CB_state.startswith("CB") and CB_state.endswith(".xls"):
            cb_number = CB_state[2:-4]  # Extract number between "CB" and ".xls"
        else:
            raise ValueError(f"CB_state must be in format 'CB#.xls' (e.g., 'CB1.xls'), got: {CB_state}")
    else:
        cb_number = str(CB_state)

    # Construct switch file path (CB_state already includes the full filename like "CB1.xls")
    switch_file = grading_dir / "input_data" / CB_state



    # Step 2: Generate ground truth tree using new_leaf_structure_tree_v2.py
    run_tree_generation(
        CB_state=cb_number,
        bus123_only_switch=switch_file,
        source_node=source_node,
        sensor_locations=sensor_locations,
        probe_depth=probe_depth
    )

    # The output JSON path from run_tree_generation
    ground_truth_path = Path(__file__).resolve().parent / "data_to_tree" / "temp_json" / f"GT_Tree_CB{cb_number}.json"


    # Step 3: Validate test tree path exists
    test_tree_path = Path(test_tree_path)
    if not test_tree_path.exists():
        raise FileNotFoundError(f"Test tree file not found: {test_tree_path}")

    # Step 4: Set output directory for metrics
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

   

    # Import the GUI-ready scripts
    from Grading_Ground_Truth.data_to_tree import RF
    from Grading_Ground_Truth.data_to_tree import TED
    from Grading_Ground_Truth.data_to_tree import SLD
    from Grading_Ground_Truth.data_to_tree import Grid_Visual
    from Grading_Ground_Truth.data_to_tree import Tree_Visual

    # Update output_dir in each module
    RF.output_dir = output_dir
    TED.output_dir = output_dir
    SLD.output_dir = output_dir
    Grid_Visual.output_dir = output_dir
    Tree_Visual.output_dir = output_dir

    # Pass full paths to the GUI scripts (not just filenames)
    # Run all metrics and always generate images
    RF.main(
        generate_image=True,
        GT_Tree=str(ground_truth_path),
        Test_Tree=str(test_tree_path)
    )

    TED.main(
        generate_image=True,
        GT_Tree=str(ground_truth_path),
        Test_Tree=str(test_tree_path)
    )

    SLD.main(
        generate_image=True,
        GT_Tree=str(ground_truth_path),
        Test_Tree=str(test_tree_path)
    )

    # Run Grid and Tree visualizations
    Grid_Visual.main(
        source_node=source_node,
        sensor_locations=sensor_locations,
        switch_file_path=switch_file,
        generate_image=True,
        probe_depth=probe_depth
    )

    Tree_Visual.main(
        source_node=source_node,
        sensor_locations=sensor_locations,
        switch_file_path=switch_file,
        generate_image=True,
        probe_depth=probe_depth
    )

    # Step 7: Read and return the combined metrics
    import json
    metrics_path = output_dir / "Metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics


if __name__ == "__main__":
    # Example usage
    example_source_node = 150
    example_sensors = [103, 41,75,  6, 16, 4,  24, 32,  50, 30, 66,  100,  85]
    example_cb_state = "CB3.xls"
    example_test_tree = grading_dir / "model_data" / "MTGP_CB3.json"
    example_probe_depth = 1

    # Run with all visualizations
    metrics = main(
        source_node=example_source_node,
        sensor_locations=example_sensors,
        CB_state=example_cb_state,
        test_tree_path=example_test_tree,
        probe_depth=example_probe_depth
    )
