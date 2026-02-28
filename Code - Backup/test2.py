# WORKING CODE


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
import numpy as np
from sklearn.metrics import roc_auc_score

# Custom Tee class to capture stdout and stderr to file and terminal
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

# Set up logging
def setup_logging(model_name, timestamp):
    log_file = f"{model_name}_{timestamp}_terminal_output.txt"
    return Tee(log_file)

# Function to count class instances from YOLO label files
def count_instances_in_labels(label_dir, class_names):
    counter = Counter()
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file)) as f:
                for line in f:
                    if line.strip():
                        class_id = int(float(line.split()[0]))
                        name = class_names.get(class_id, f"class_{class_id}")
                        counter[name] += 1
    return dict(counter)

# Function to print and log a summary table of class counts
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
            print(f"Directory '{d}' does not exist.")
            sys.exit(1)
        if len(os.listdir(d)) == 0:
            print(f"Directory '{d}' is empty.")
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

# Function to compute Jaccard Index (IoU) for predictions
def compute_jaccard(pred_boxes, gt_boxes):
    if not pred_boxes or not gt_boxes:
        return 0.0
    inter_xmin = max(pred_boxes[0], gt_boxes[0])
    inter_ymin = max(pred_boxes[1], gt_boxes[1])
    inter_xmax = min(pred_boxes[2], gt_boxes[2])
    inter_ymax = min(pred_boxes[3], gt_boxes[3])
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    pred_area = (pred_boxes[2] - pred_boxes[0]) * (pred_boxes[3] - pred_boxes[1])
    gt_area = (gt_boxes[2] - gt_boxes[0]) * (gt_boxes[3] - gt_boxes[1])
    union_area = pred_area + gt_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# Function to compute AUC for a single class
def compute_class_auc(predictions, ground_truths, class_id, num_instances):
    y_true = []
    y_score = []
    for pred, gt in zip(predictions, ground_truths):
        gt_class = gt[0] if gt else -1
        pred_class = pred[5] if pred else -1
        pred_conf = pred[4] if pred else 0.0
        y_true.append(1 if gt_class == class_id else 0)
        y_score.append(pred_conf if pred_class == class_id else 0.0)
    if len(y_true) != len(y_score):
        print(f"Warning: Mismatched lengths for class {class_id}: y_true={len(y_true)}, y_score={len(y_score)}")
        return np.nan
    if len(set(y_true)) < 2 or sum(y_true) == 0:
        print(f"Warning: Insufficient positive samples for AUC computation for class {class_id}")
        return np.nan
    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"AUC for class {class_id} ({class_names[class_id]}): {auc:.4f}")
        return auc
    except ValueError as e:
        print(f"Error computing AUC for class {class_id}: {e}")
        return np.nan

# Function to save and log confusion matrix
def save_confusion_matrix(cm, class_names, output_path, split_name):
    try:
        matrix = cm.matrix[:-1, :-1]  # Exclude background class
        df_cm = pd.DataFrame(matrix, index=class_names, columns=class_names)
        df_cm.to_csv(output_path, index=True)
        print(f"\n{split_name} Confusion Matrix:")
        print("-" * 50)
        print(df_cm.to_string())
        print(f"Confusion matrix saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save {split_name} confusion matrix: {e}")

def main():
    data_yaml_path = r"D:\2.3 Code_s\RF-Python - Copy-27Feb\Sagittal_T2_515_384x384\Split_Dataset\data.yaml"
    os.chdir(os.path.dirname(data_yaml_path))
    project_path = os.getcwd()
    print("Working directory set to:", project_path)

    model_name = "yolo12m"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _ = setup_logging(model_name, timestamp)

    data_yaml = update_yaml_path(data_yaml_path)
    class_names_dict = {i: name for i, name in enumerate(data_yaml['names'])}
    global class_names
    class_names = list(class_names_dict.values())

    train_img_dir = os.path.join(project_path, "train", "images")
    val_img_dir   = os.path.join(project_path, "val", "images")
    test_img_dir  = os.path.join(project_path, "test", "images")

    train_label_dir = os.path.join(project_path, "train", "labels")
    val_label_dir   = os.path.join(project_path, "val", "labels")
    test_label_dir  = os.path.join(project_path, "test", "labels")

    check_dataset_dirs(train_img_dir, val_img_dir, test_img_dir, train_label_dir, val_label_dir, test_label_dir)

    model = YOLO('yolo12m.pt')
    num_classes = data_yaml.get('nc')
    if num_classes is None:
        print("The key 'nc' (number of classes) is missing in data.yaml.")
        sys.exit(1)

    # Log class distribution
    train_instances = count_instances_in_labels(train_label_dir, class_names_dict)
    val_instances   = count_instances_in_labels(val_label_dir, class_names_dict)
    test_instances  = count_instances_in_labels(test_label_dir, class_names_dict)
    print_summary_table(train_instances, "Training")
    print_summary_table(val_instances, "Validation")
    print_summary_table(test_instances, "Test")

    abs_data_yaml = os.path.abspath(data_yaml_path)
    run_name = f'{model_name}_{timestamp}'
    epochs = 20  # Set to 200 for full training

    # Training
    start_time = time.time()
    results = model.train(
        data=abs_data_yaml,
        project=project_path,
        name=run_name,
        epochs=epochs,
        batch=2,
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
    avg_epoch_time = total_time / epochs if epochs > 0 else total_time

    print(f"\nTotal Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f} seconds")

    # Save training and validation logs
    results_csv = os.path.join(project_path, run_name, "results.csv")
    if os.path.exists(results_csv):
        df_train = pd.read_csv(results_csv)
        train_log_path = os.path.join(project_path, f"{run_name}_training.csv")
        val_log_path = os.path.join(project_path, f"{run_name}_validation.csv")
        df_train.to_csv(train_log_path, index=False)
        df_val = df_train[['epoch', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].copy()
        df_val.to_csv(val_log_path, index=False)
        print(f"Training log saved to: {train_log_path}")
        print(f"Validation log saved to: {val_log_path}")
        print("\nTraining Log Preview:")
        print(df_train.head().to_string())

    # Display training results plot non-blocking
    results_plot_path = os.path.join(project_path, run_name, "results.png")
    if os.path.exists(results_plot_path):
        plt.ion()  # Enable interactive mode for non-blocking display
        img = plt.imread(results_plot_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title("Training Results")
        plt.axis('off')
        plt.show()
        print("Training results plot displayed. Close the plot window to continue.")

    # Compute and save confusion matrices
    print("\nComputing Training Confusion Matrix...")
    try:
        train_results = model.val(
            data=abs_data_yaml,
            split="train",
            project=project_path,
            name=f"{run_name}_train_eval",
            plots=True
        )
        train_cm_path = os.path.join(project_path, f"{run_name}_training_confusion_matrix.csv")
        save_confusion_matrix(train_results.confusion_matrix, class_names, train_cm_path, "Training")
    except Exception as e:
        print(f"Failed to compute training confusion matrix: {e}")

    print("\nComputing Validation Confusion Matrix...")
    try:
        val_results = model.val(
            data=abs_data_yaml,
            split="val",
            project=project_path,
            name=f"{run_name}_val_eval",
            plots=True
        )
        val_cm_path = os.path.join(project_path, f"{run_name}_validation_confusion_matrix.csv")
        save_confusion_matrix(val_results.confusion_matrix, class_names, val_cm_path, "Validation")
    except Exception as e:
        print(f"Failed to compute validation confusion matrix: {e}")

    # Test evaluation with inference time
    print("\nEvaluating on Test Set...")
    test_start_time = time.time()
    try:
        test_results = model.val(
            data=abs_data_yaml,
            split="test",
            project=project_path,
            name=f"{run_name}_test",
            plots=True
        )
        test_inference_time = time.time() - test_start_time
        print(f"Test Inference Time: {test_inference_time:.2f} seconds")
        test_cm_path = os.path.join(project_path, f"{run_name}_testing_confusion_matrix.csv")
        save_confusion_matrix(test_results.confusion_matrix, class_names, test_cm_path, "Testing")
    except Exception as e:
        print(f"Failed to compute test confusion matrix: {e}")
        test_inference_time = 0

    # Save test metrics with additional metrics
    test_metrics_path = os.path.join(project_path, f"{run_name}_test", "test_metrics.csv")
    try:
        # Extract class-wise metrics from test_results
        rows = []
        all_predictions = []
        all_ground_truths = []
        for img_path in glob.glob(os.path.join(test_img_dir, "*.png")):
            predictions = []
            ground_truths = []
            results = model.predict(img_path, conf=0.25, verbose=False)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(test_label_dir, f"{base_name}.txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            class_id = int(float(parts[0]))
                            x_center, y_center, w, h = map(float, parts[1:5])
                            x1 = (x_center - w / 2) * 384
                            y1 = (y_center - h / 2) * 384
                            x2 = (x_center + w / 2) * 384
                            y2 = (y_center + h / 2) * 384
                            ground_truths.append([class_id, x1, y1, x2, y2])
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    class_id = int(box.cls.item())
                    predictions.append([x1, y1, x2, y2, conf, class_id])
            # Pad predictions and ground truths to ensure equal length
            max_len = max(len(predictions), len(ground_truths))
            predictions.extend([None] * (max_len - len(predictions)))
            ground_truths.extend([None] * (max_len - len(ground_truths)))
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)

        jaccard_scores = []
        for pred, gt in zip(all_predictions, all_ground_truths):
            if pred and gt and pred[5] == gt[0]:
                pred_box = pred[:4]
                gt_box = gt[1:]
                iou = compute_jaccard(pred_box, gt_box)
                jaccard_scores.append(iou)
        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0

        auc_scores = {}
        for i in range(num_classes):
            auc = compute_class_auc(all_predictions, all_ground_truths, i, num_classes)
            auc_scores[class_names[i]] = auc
        overall_auc = np.nanmean(list(auc_scores.values())) if auc_scores else np.nan

        # Extract class-wise metrics from test_results
        for i, name in enumerate(class_names):
            precision = test_results.box.p[i] if i < len(test_results.box.p) else 0.0
            recall = test_results.box.r[i] if i < len(test_results.box.r) else 0.0
            map50 = test_results.box.ap50[i] if i < len(test_results.box.ap50) else 0.0
            map50_95 = test_results.box.ap[i] if i < len(test_results.box.ap) else 0.0
            rows.append({
                "Class": name,
                "Precision": precision,
                "Recall": recall,
                "mAP50": map50,
                "mAP50-95": map50_95,
                "Jaccard": avg_jaccard,
                "AUC": auc_scores.get(name, np.nan)
            })

        df_test = pd.DataFrame(rows)
        df_test.to_csv(test_metrics_path, index=False)
        print(f"Test metrics saved to: {test_metrics_path}")
        print("\nTest Metrics Preview:")
        print(df_test.to_string())

        test_log_path = os.path.join(project_path, f"{run_name}_testing.csv")
        df_test.to_csv(test_log_path, index=False)
        print(f"Testing log saved to: {test_log_path}")
    except Exception as e:
        print(f"Failed to compute test metrics: {e}")

    # Archive metrics
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
        print(f"Could not append metrics to archive: {e}")

    # Log additional metrics
    print(f"\nAdditional Metrics:")
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Average Jaccard Index: {avg_jaccard:.4f}")

    # Run predictions on test set and save to CSV
    print("\nPredicting all test images and saving results...")
    pred_output_dir = os.path.join(project_path, "predicted_images", "predict")
    os.makedirs(pred_output_dir, exist_ok=True)
    pred_csv_path = os.path.join(pred_output_dir, "predictions.csv")
    pred_rows = []
    for img_path in glob.glob(os.path.join(test_img_dir, "*.png")):
        results = model.predict(img_path, conf=0.25, verbose=False)
        class_counts = Counter()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = class_names[class_id]
                class_counts[class_name] += 1
        row = {"Image": os.path.basename(img_path)}
        row.update({name: class_counts.get(name, 0) for name in class_names})
        pred_rows.append(row)
    df_pred = pd.DataFrame(pred_rows)
    df_pred.to_csv(pred_csv_path, index=False)
    print(f"All predictions saved to: {pred_csv_path}")

    # Export model
    try:
        model.export(format="onnx")
        print("Model exported in ONNX format.")
    except Exception as e:
        print(f"Failed to export model to ONNX: {e}")

if __name__ == '__main__':
    main()