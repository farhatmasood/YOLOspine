# training.py
import os
from ultralytics import YOLO
from Code.config import MODEL_PATH, DATA_YAML_PATH, PROJECT_PATH, EPOCHS, BATCH_SIZE, IMAGE_SIZE
from Code.dataset_utils import update_yaml_path, check_dataset_dirs, count_instances_in_labels, print_summary_table

def run_training_pipeline():
    print(f"Working directory set to: {PROJECT_PATH}")

    # Step 1: Update and validate dataset YAML file
    data_yaml = update_yaml_path(DATA_YAML_PATH)
    train_img_dir = data_yaml['train'].replace('/images', '/images')  # Ensure correct path format
    val_img_dir = data_yaml['val'].replace('/images', '/images')
    test_img_dir = data_yaml['test'].replace('/images', '/images')

    # Check if dataset directories exist and are not empty
    check_dataset_dirs(train_img_dir, val_img_dir, test_img_dir)

    # Step 2: Count object instances in each dataset split
    model = YOLO(MODEL_PATH)  # Load YOLO model
    train_label_dir = train_img_dir.replace("images", "labels")
    val_label_dir = val_img_dir.replace("images", "labels")
    test_label_dir = test_img_dir.replace("images", "labels")

    train_summary = count_instances_in_labels(train_label_dir, model.names)
    val_summary = count_instances_in_labels(val_label_dir, model.names)
    test_summary = count_instances_in_labels(test_label_dir, model.names)

    # Print a summary table showing class-wise instance counts
    print_summary_table(train_summary, "Training")
    print_summary_table(val_summary, "Validation")
    print_summary_table(test_summary, "Test")

    # Step 3: Run training
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        project=PROJECT_PATH,
        name=os.path.basename(PROJECT_PATH),
        deterministic=True,  # Ensure reproducibility
        val=True  # Enable validation during training
    )

    # Step 4: Export trained model to ONNX format for interoperability
    model.export(format="onnx")
