# evaluation.py
import os
import pandas as pd
from ultralytics import YOLO
from Code.config import MODEL_PATH, DATA_YAML_PATH, PROJECT_PATH, CONF_THRESHOLD


def evaluate_model():
    """
    Loads a trained YOLO model, evaluates it on the test set,
    logs key metrics and generates plots (including confusion matrix).
    """
    print("Loading trained model for evaluation...")
    model = YOLO(MODEL_PATH)

    print(f"\nEvaluating on test set with confidence threshold: {CONF_THRESHOLD}")
    results = model.val(
        data=DATA_YAML_PATH,
        split="test",
        conf=CONF_THRESHOLD,
        project=PROJECT_PATH,
        name="test_eval",
        plots=True  # This includes the confusion matrix, PR curve, etc.
    )

    # Save metrics to CSV
    try:
        metrics_dict = results.results_dict
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

        df = pd.DataFrame(rows)
        metrics_csv_path = os.path.join(PROJECT_PATH, "test_metrics.csv")
        df.to_csv(metrics_csv_path, index=False)
        print(f"Test metrics saved to: {metrics_csv_path}")

    except Exception as e:
        print("Could not save test metrics CSV. Reason:", e)

    print("Evaluation completed.")
