# Working but Logs Not Working, Prediction Images WORKING


# Environment variable setup to ensure the script runs correctly
import os
os.environ.pop("YOLO_DATA", None)
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
import glob
import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO
from datetime import datetime
import time
import pandas as pd
from collections import Counter

# ✅ New function to count class instances from YOLO label files
def count_instances_in_labels(label_dir, class_names):
    counter = Counter()
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file)) as f:
                for line in f:
                    if line.strip():
                        class_id = int(float(line.split()[0]))  # ✅ handles '1.0' as well
                        name = class_names.get(class_id, f"class_{class_id}")
                        counter[name] += 1
    return dict(counter)

# Function to print a summary table of class counts
def print_summary_table(summary, set_name):
    print(f"\n{set_name} Set Summary (Instances per class):")
    print("-" * 40)
    total = 0
    for cls, count in summary.items():
        print(f"{cls:25s} : {count}")
        total += count
    print(f"{'Total':25s} : {total}")
    print("-" * 40)

# Function to validate dataset directories
def check_dataset_dirs(*dirs):
    for d in dirs:
        if not os.path.exists(d):
            print(f"Error: Directory '{d}' does not exist.")
            sys.exit(1)
        if len(os.listdir(d)) == 0:
            print(f"Error: Directory '{d}' is empty.")
            sys.exit(1)

def update_yaml_path(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if data.get("path", "") != ".":
        print("Updating 'path' in data.yaml to '.'")
        data["path"] = "."
    for key in ["train", "val", "test"]:
        if key in data and isinstance(data[key], str):
            data[key] = data[key].replace("\\", "/")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return data

def main():
    data_yaml_path = r"D:\\2.3 Code_s\\RF-Python - Copy-27Feb\\Sagittal_T2_515_384x384\\Split_Dataset\\data.yaml"
    os.chdir(os.path.dirname(data_yaml_path))
    project_path = os.getcwd()
    print("Working directory set to:", project_path)

    data_yaml = update_yaml_path(data_yaml_path)
    class_names = {i: name for i, name in enumerate(data_yaml['names'])}  # ✅ use names from data.yaml

    train_img_dir = os.path.join(project_path, "train", "images")
    val_img_dir   = os.path.join(project_path, "val", "images")
    test_img_dir  = os.path.join(project_path, "test", "images")

    train_label_dir = os.path.join(project_path, "train", "labels")
    val_label_dir   = os.path.join(project_path, "val", "labels")
    test_label_dir  = os.path.join(project_path, "test", "labels")

    check_dataset_dirs(train_img_dir, val_img_dir, train_label_dir, val_label_dir)

    model = YOLO('yolo12m.pt')
    num_classes = data_yaml.get('nc')
    if num_classes is None:
        print("Error: The key 'nc' (number of classes) is missing in data.yaml.")
        sys.exit(1)

    # ✅ Print true class distribution from labels using corrected class names
    train_instances = count_instances_in_labels(train_label_dir, class_names)
    val_instances   = count_instances_in_labels(val_label_dir, class_names)
    test_instances  = count_instances_in_labels(test_label_dir, class_names)
    print_summary_table(train_instances, "Training")
    print_summary_table(val_instances, "Validation")
    print_summary_table(test_instances, "Test")

    abs_data_yaml = os.path.abspath(data_yaml_path)
    run_name = f'yolo12m_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    start_time = time.time()
    results = model.train(
        data=abs_data_yaml,
        project=project_path,
        name=run_name,
        epochs=20,
        batch=32,
        imgsz=384,
        degrees=7,
        translate=0.1,
        scale=0.3,
        hsv_s=0.3,
        hsv_v=0.4,
        mosaic=0.7,
        mixup=0.1
    )

    total_time = time.time() - start_time
    avg_epoch_time = total_time / 200

    print(f"\nTotal Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f} seconds")

    results_csv = os.path.join(project_path, run_name, "results.csv")
    if os.path.exists(results_csv):
        print(f"Training log saved to: {results_csv}")
        df_train = pd.read_csv(results_csv)
        print(df_train.head())

    results_plot_path = os.path.join(project_path, run_name, "results.png")
    if os.path.exists(results_plot_path):
        img = plt.imread(results_plot_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title("Training Results")
        plt.axis('off')
        plt.show()

    print("\nEvaluating on Test Set...")
    test_results = model.val(
        data=abs_data_yaml,
        split="test",
        project=project_path,
        name=f"{run_name}_test",
        plots=True
    )

    test_metrics_path = os.path.join(project_path, f"{run_name}_test", "test_metrics.csv")
    try:
        metrics_dict = test_results.results_dict
        class_names = model.names

        rows = []
        for i, name in class_names.items():
            rows.append({
                "Class": name,
                "Precision": metrics_dict["metrics/precision"][i],
                "Recall": metrics_dict["metrics/recall"][i],
                "mAP50": metrics_dict["metrics/mAP50"][i],
                "mAP50-95": metrics_dict["metrics/mAP50-95"][i]
            })

        df_test = pd.DataFrame(rows)
        df_test.to_csv(test_metrics_path, index=False)
        print(f"Test metrics saved to: {test_metrics_path}")
    except Exception as e:
        print("Test metrics not available.")
        print("Details:", e)

    metrics_archive_path = os.path.join(project_path, "metrics_log.csv")
    try:
        if os.path.exists(metrics_archive_path):
            df_archive = pd.read_csv(metrics_archive_path)
            df_combined = pd.concat([df_archive, df_test], ignore_index=True)
        else:
            df_combined = df_test
        df_combined.to_csv(metrics_archive_path, index=False)
        print(f"All test metrics appended to: {metrics_archive_path}")
    except Exception as e:
        print("Could not append metrics to archive.")
        print("Details:", e)

    # ✅ Run full prediction on test set and save images
    print("\nPredicting all test images and saving results...")
    pred_output_dir = os.path.join(project_path, "predicted_images")
    model.predict(
        source=test_img_dir,
        project=pred_output_dir,
        name="predict",
        save=True,
        conf=0.25
    )
    print(f"All predictions saved to: {os.path.join(pred_output_dir, 'predict')}")

    # Export model
    model.export(format="onnx")
    print("Model exported in ONNX format.")

if __name__ == '__main__':
    main()
