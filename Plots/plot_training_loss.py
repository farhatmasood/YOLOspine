# This script reads training data from a CSV file and generates three plots:
# 1. Training Loss vs. Epochs:
#    - Plots the training loss curves for 'train/box_loss', 'train/cls_loss', and 'train/dfl_loss' against epochs.
# 2. Metrics vs. Epochs:
#    - Plots the performance metrics ('metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', and 'metrics/mAP50-95(B)') against epochs.
# 3. Precision-Recall Curve:
#    - Plots the precision-recall curve using 'metrics/recall(B)' and 'metrics/precision(B)'.
# The script uses Matplotlib to create the plots and customizes them with titles, labels, legends, and grids.
# The plots are displayed interactively using `plt.show()`.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the training data
input_path = 'yolo_metrics_training_data.csv'
df = pd.read_csv(input_path)

# # Plot training loss vs. epochs
# plt.figure(figsize=(10, 6))
# plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='blue')
# plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='orange')
# plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', color='green')

# # Customize the plot
# plt.title('Training Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend(title='Loss Type')
# plt.grid(True)
# plt.tight_layout()

# # Show the plot
# plt.show()

# Plot metrics vs. epochs
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='blue')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='orange')
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='green')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='red')

# Customize the plot
plt.title('Metrics vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend(title='Metrics')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# Since 'predicted_confidence' and 'ground_truth_label' are not available, use metrics/precision(B) and metrics/recall(B)
# directly to plot the Precision-Recall curve as an approximation.

# Plot Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(df['metrics/recall(B)'], df['metrics/precision(B)'], label='Precision-Recall Curve', color='purple')

# Customize the plot
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(title='Curve')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
