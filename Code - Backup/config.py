# config.py
from datetime import datetime
import os

# Paths
DATA_YAML_PATH = r"D:\\2.3 Code_s\\RF-Python - Copy-27Feb\\Sagittal_T2_515_384x384\\Split_Dataset\\data.yaml"
PROJECT_PATH = os.path.dirname(DATA_YAML_PATH)
TEST_IMG_DIR = os.path.join(PROJECT_PATH, "test", "images")

# Model
MODEL_NAME = 'yolo11m.pt'
CONF_THRESHOLD = 0.25
EPOCHS = 50
IMG_SIZE = 384

# Output
RUN_NAME = f"yolo11m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = os.path.join(PROJECT_PATH, RUN_NAME)
