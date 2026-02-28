# prediction.py
import os
from ultralytics import YOLO
from Code.config import MODEL_PATH, TEST_IMAGE_DIR, PROJECT_PATH, CONF_THRESHOLD


def run_prediction():
    """
    Perform prediction using the trained YOLO model on the test image directory.
    Saves the prediction results including bounding boxes and confidence values.
    """
    print("\nStarting test set predictions...")

    # Load the trained YOLO model from the given path
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    # Define directory to save predictions
    predict_out_dir = os.path.join(PROJECT_PATH, f"predictions_conf{int(CONF_THRESHOLD * 100)}")

    # Run prediction on all test images
    results = model.predict(
        source=TEST_IMAGE_DIR,        # Directory of test images
        project=predict_out_dir,      # Output directory for predictions
        name="predict",              # Subfolder name inside project
        save=True,                    # Save images with predictions
        conf=CONF_THRESHOLD,          # Classification confidence threshold
        save_txt=True,                # Save label files in YOLO format
        save_conf=True                # Save confidence scores in label files
    )

    print("Prediction completed.")
    print(f"Results saved in: {os.path.join(predict_out_dir, 'predict')}")
