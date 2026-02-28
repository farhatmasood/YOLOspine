import pandas as pd
import matplotlib.pyplot as plt

# Load specific columns from Excel and specify the sheet name
df = pd.read_excel("GPU.xlsx", sheet_name="GPU-Train (2)", usecols="B,C,D,E,F,G")

# Rename columns for clarity
df.columns = ['YOLOv12', 'YOLOv11', 'YOLOv10', 'YOLOv9', 'YOLOv8', 'YOLOs']

# Convert all columns to numeric (force errors to NaN)
df = df.apply(pd.to_numeric, errors='coerce')

# Prepare data for box plot (drop NaN from each column)
gpu_data = [df[col].dropna() for col in df.columns]

# Plot
plt.figure(figsize=(10, 6))
box = plt.boxplot(
    gpu_data,
    patch_artist=False,  # Disable custom box colors (no fill)
    tick_labels=df.columns  # Use tick_labels instead of labels
)

# Customize box outlines
for i, box_element in enumerate(box['boxes']):
    if df.columns[i] == 'YOLOs':
        box_element.set_color('red')  # Orange outline for YOLOs
    else:
        box_element.set_color('black')  # Black outline for other models

# Add grid lines for better readability
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Add labels and styling
plt.ylabel("GPU Usage (GB)", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
