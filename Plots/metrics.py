import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file and get all sheet names
file_path = "Training.xlsx"
excel_data = pd.ExcelFile(file_path)
sheet_names = excel_data.sheet_names  # Get sheet titles
print("Sheet Titles:", sheet_names)

# Filter sheets from YOLO12 to YOLO8
selected_sheets = [sheet for sheet in sheet_names if sheet in ["YOLO12", "YOLO11", "YOLO10", "YOLO9", "YOLO8"]]
print("Selected Sheets:", selected_sheets)

# # Initialize a plot for train/box_loss
# plt.figure(figsize=(10, 6))

# # Process each selected sheet
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['train/box_loss'], label=sheet)

# # Customize the plot
# plt.title("Train/Box Loss for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Train/Box Loss")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for train/cls_loss
# plt.figure(figsize=(10, 6))

# # Process each selected sheet
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['train/cls_loss'], label=sheet)

# # Customize the plot
# plt.title("Train/Cls Loss for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Train/Cls Loss")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for train/dfl_loss
# plt.figure(figsize=(10, 6))

# # Process each selected sheet
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['train/dfl_loss'], label=sheet)

# # Customize the plot
# plt.title("Train/DFL Loss for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Train/DFL Loss")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for metrics/precision(B)
# plt.figure(figsize=(10, 6))

# # Process each selected sheet
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['metrics/precision(B)'], label=sheet)

# # Customize the plot
# plt.title("Metrics/Precision(B) for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Metrics/Precision(B)")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for metrics/recall(B)
# plt.figure(figsize=(10, 6))

# # Process each selected sheet
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['metrics/recall(B)'], label=sheet)

# # Customize the plot
# plt.title("Metrics/Recall(B) for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Metrics/Recall(B)")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for metrics/mAP50(B)
# plt.figure(figsize=(10, 6))
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['metrics/mAP50(B)'], label=sheet)
# plt.title("Metrics/mAP50(B) for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Metrics/mAP50(B)")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for metrics/mAP50-95(B)
# plt.figure(figsize=(10, 6))
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['metrics/mAP50-95(B)'], label=sheet)
# plt.title("Metrics/mAP50-95(B) for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Metrics/mAP50-95(B)")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for val/box_loss
# plt.figure(figsize=(10, 6))
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['val/box_loss'], label=sheet)
# plt.title("Validation Box Loss for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Validation Box Loss")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for val/cls_loss
# plt.figure(figsize=(10, 6))
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['val/cls_loss'], label=sheet)
# plt.title("Validation Class Loss for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Validation Class Loss")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for val/dfl_loss
# plt.figure(figsize=(10, 6))
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['val/dfl_loss'], label=sheet)
# plt.title("Validation DFL Loss for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Validation DFL Loss")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for lr/pg0
# plt.figure(figsize=(10, 6))
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['lr/pg0'], label=sheet)
# plt.title("Learning Rate (pg0) for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Learning Rate (pg0)")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for lr/pg1
# plt.figure(figsize=(10, 6))
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['lr/pg1'], label=sheet)
# plt.title("Learning Rate (pg1) for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Learning Rate (pg1)")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Initialize a plot for lr/pg2
# plt.figure(figsize=(10, 6))
# for sheet in selected_sheets:
#     data = pd.read_excel(file_path, sheet_name=sheet)
#     plt.plot(data['epoch'], data['lr/pg2'], label=sheet)
# plt.title("Learning Rate (pg2) for YOLO Models")
# plt.xlabel("Epochs")
# plt.ylabel("Learning Rate (pg2)")
# plt.legend(title="Models")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Initialize a plot for Precision-Recall Curve
plt.figure(figsize=(10, 6))

# Process each selected sheet
for sheet in selected_sheets:
    data = pd.read_excel(file_path, sheet_name=sheet)
    plt.plot(data['metrics/recall(B)'], data['metrics/precision(B)'], label=sheet)

# Customize the plot
plt.title("Precision-Recall Curve for YOLO Models")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(title="Models")
plt.grid(True)
plt.tight_layout()
plt.show()
