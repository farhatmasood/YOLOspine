# CREATES A CSV FILE FROM YOLO TRAINING LOGS.
# This script processes a YOLO training log file and extracts relevant metrics.
# It then saves the extracted data into a CSV file for further analysis.    
# GPU usage and loss metrics are averaged over epochs.

import re
import numpy as np
import csv

log_file_path = "12\yolo12m_20250418_133723_terminal_output.txt"
output_csv_path = "12\yolo12_GPUs.csv"

# Regex to match training lines (handles variable decimal places)
epoch_pattern = re.compile(r"(\d+)/200\s+(\d+\.\d+)G\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+):.*\d+%.*\|\s*\d+/23")

epoch_data = {}
try:
    with open(log_file_path, 'r', encoding='utf-8') as file:
        epoch = None
        for line in file:
            epoch_match = epoch_pattern.search(line)

            if epoch_match:
                try:
                    epoch = int(epoch_match.group(1))
                    gpu_memory = float(epoch_match.group(2))
                    box_loss = float(epoch_match.group(3))
                    cls_loss = float(epoch_match.group(4))
                    dfl_loss = float(epoch_match.group(5))
                    gpu_usage_mb = int(epoch_match.group(6))

                    if epoch not in epoch_data:
                        epoch_data[epoch] = {
                            "box_loss": [],
                            "cls_loss": [],
                            "dfl_loss": [],
                            "gpu_usage": [],
                            "gpu_memory": gpu_memory
                        }

                    epoch_data[epoch]["box_loss"].append(box_loss)
                    epoch_data[epoch]["cls_loss"].append(cls_loss)
                    epoch_data[epoch]["dfl_loss"].append(dfl_loss)
                    epoch_data[epoch]["gpu_usage"].append(gpu_usage_mb)
                except ValueError as e:
                    print(f"Error parsing epoch line: {line.strip()} - {e}")

except FileNotFoundError:
    print(f"Error: Log file not found at {log_file_path}")
    exit(1)
except UnicodeDecodeError:
    print(f"Error: Unable to decode {log_file_path} with UTF-8")
    exit(1)

# Write to CSV with reduced headers
try:
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Epoch", "GPU Usage (GB)", "Peak GPU Usage (MB)", "Avg GPU Usage (MB)",
            "Box Loss", "Cls Loss", "DFL Loss"
        ])

        for epoch in sorted(epoch_data.keys()):
            data = epoch_data[epoch]
            avg_box_loss = np.mean(data["box_loss"])
            avg_cls_loss = np.mean(data["cls_loss"])
            avg_dfl_loss = np.mean(data["dfl_loss"])
            peak_gpu_usage = np.max(data["gpu_usage"])
            avg_gpu_usage = np.mean(data["gpu_usage"])

            writer.writerow([
                epoch, data["gpu_memory"], peak_gpu_usage, round(avg_gpu_usage, 2),
                round(avg_box_loss, 4), round(avg_cls_loss, 4), round(avg_dfl_loss, 4)
            ])

    print(f"✅ Summary saved to: {output_csv_path}")

except IOError as e:
    print(f"Error writing to CSV: {e}")
    exit(1)