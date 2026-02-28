# ONLY PREDICTION USING THE SAVED MODEL


import os
import sys
import yaml
import time
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from matplotlib import pyplot as plt

# Configuration
MODEL_PATH = r"D:\2.3 Code_s\RF-Python - Copy-27Feb\Sagittal_T2_515_384x384\Split_Dataset\yolo11l_20250327_113156\weights\best.pt"
DATA_YAML_PATH = r"D:\2.3 Code_s\RF-Python - Copy-27Feb\Sagittal_T2_515_384x384\Split_Dataset\data.yaml"
TEST_IMG_DIR = r"D:\2.3 Code_s\RF-Python - Copy-27Feb\Sagittal_T2_515_384x384\Split_Dataset\test\images"
PROJECT_PATH = os.path.dirname(DATA_YAML_PATH)
CONF_THRESHOLD = 0.75  # Adjustable confidence threshold

def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    # Run evaluation and generate confusion matrix
    print("\nEvaluating on test set with confidence threshold:", CONF_THRESHOLD)
    test_results = model.val(
        data=DATA_YAML_PATH,
        split="test",
        conf=CONF_THRESHOLD,
        project=PROJECT_PATH,
        name="yolo11l_test_eval",
        plots=True
    )

    # Make predictions on test set (to save images)
    predict_out_dir = os.path.join(PROJECT_PATH, f"predictions_conf{int(CONF_THRESHOLD*100)}")
    results = model.predict(
        source=TEST_IMG_DIR,
        project=predict_out_dir,
        name="predict",
        save=True,
        conf=CONF_THRESHOLD,
        save_txt=True,
        save_conf=True
    )

    print("Prediction and evaluation complete.")
    print(f"Predictions saved to: {os.path.join(predict_out_dir, 'predict')}")

if __name__ == '__main__':
    main()
