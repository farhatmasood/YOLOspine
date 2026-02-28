# dataset_utils.py
import os
import sys
import yaml
from collections import defaultdict


def update_yaml_path(yaml_path):
    """
    Updates the 'path' field in the dataset YAML file to the current directory.
    Also replaces backslashes with forward slashes in path fields for compatibility.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if data.get("path", "") != ".":
        print("Updating 'path' in data.yaml to '.'")
        data["path"] = "."
    for key in ["train", "val", "test"]:
        if key in data and isinstance(data[key], str):
            data[key] = data[key].replace("\\", "/")  # Normalize path separators
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return data


def check_dataset_dirs(*dirs):
    """
    Validates that specified dataset directories exist and are not empty.
    Terminates the program if any directory is missing or empty.
    """
    for d in dirs:
        if not os.path.exists(d):
            print(f"Error: Directory '{d}' does not exist.")
            sys.exit(1)
        if len(os.listdir(d)) == 0:
            print(f"Error: Directory '{d}' is empty.")
            sys.exit(1)


def count_instances_in_labels(label_dir, class_names):
    """
    Counts the number of instances per class from YOLO-format label text files.

    Args:
        label_dir (str): Directory containing .txt annotation files.
        class_names (dict): Mapping from class indices to class names.

    Returns:
        dict: Summary of instances per class name.
    """
    summary = defaultdict(int)
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file)) as f:
                for line in f:
                    try:
                        class_id = int(float(line.strip().split()[0]))  # Handle int and float class IDs
                        class_name = class_names[class_id]  # Map ID to name
                        summary[class_name] += 1
                    except Exception as e:
                        print(f"Error reading {label_file}: {e}")
    return dict(summary)


def print_summary_table(summary, set_name):
    """
    Prints a formatted summary table of instance counts per class.

    Args:
        summary (dict): Class-wise instance count.
        set_name (str): Label for the dataset split (train/val/test).
    """
    print(f"\n{set_name} Set Summary (Instances per class):")
    print("-" * 40)
    total = 0
    for cls, count in summary.items():
        print(f"{cls:25s} : {count}")  # Align class names to 25-character width
        total += count
    print(f"{'Total':25s} : {total}")
    print("-" * 40)
