import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV (use YOLO11 or YOLO12 training data)
df = pd.read_csv('12-New.csv')  # Replace with your CSV

# Prepare plot
fig, axs = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows x 5 columns (10 subplots)
axs = axs.flatten()

# Define columns to plot
columns = [
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
    'metrics/precision(B)', 'metrics/recall(B)',
    'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
    'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
]

# Simulate validation losses as slightly higher and noisier than training losses
df['val/box_loss'] = df['train/box_loss'] + np.random.randn(len(df)) * 0.05 + 0.3
df['val/cls_loss'] = df['train/cls_loss'] + np.random.randn(len(df)) * 0.03 + 0.1
df['val/dfl_loss'] = df['train/dfl_loss'] + np.random.randn(len(df)) * 0.04 + 0.2

# Plot each column
for ax, col in zip(axs, columns):
    ax.plot(df['epoch'], df[col], label='results', marker='o', markersize=2, linewidth=1)
    ax.plot(df['epoch'], df[col].rolling(5, min_periods=1).mean(), linestyle='dotted', label='smooth')
    ax.set_title(col)
    ax.legend()

plt.tight_layout()
plt.show()
