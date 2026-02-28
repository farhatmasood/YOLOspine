import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Number of epochs
epochs = 200

# Provided values for first 5 epochs and last epoch
initial_data = {
    'metrics/precision(B)': [0.10344, 0.519216429, 0.5108075, 0.502398571, 0.58005],
    'metrics/recall(B)': [0.11561, 0.329066429, 0.3654675, 0.401868571, 0.47446],
    'metrics/mAP50(B)': [0.08109, 0.290007857, 0.318793214, 0.347578571, 0.4393],
    'metrics/mAP50-95(B)': [0.03646, 0.144295714, 0.159061429, 0.173827143, 0.20659]
}

final_data = {
    'metrics/precision(B)': 0.91398,
    'metrics/recall(B)': 0.944,
    'metrics/mAP50(B)': 0.94112,
    'metrics/mAP50-95(B)': 0.86262
}

# Function to generate realistic metric curve for training metrics (unchanged)
def generate_metric_curve(start_value, end_value, epochs, metric_name, initial_data, start_epoch=5, noise_scale=0.015, growth_rate=0.035):
    num_epochs = epochs - start_epoch
    t = np.arange(num_epochs)
    metric_curve = end_value / (1 + np.exp(-growth_rate * (t - num_epochs / 2)))
    metric_curve = start_value + (metric_curve - metric_curve[0]) * (end_value - start_value) / (metric_curve[-1] - metric_curve[0])
    noise = np.random.normal(0, noise_scale, num_epochs) * (1 - t / num_epochs)**0.5
    metric_curve += noise
    dip_start_1, dip_end_1 = 90 - start_epoch, 100 - start_epoch
    dip_start_2, dip_end_2 = 150 - start_epoch, 160 - start_epoch
    if dip_end_1 > 0:
        metric_curve[dip_start_1:dip_end_1] -= 0.02
    if dip_end_2 > 0:
        metric_curve[dip_start_2:dip_end_2] -= 0.015
    metric_curve = np.clip(metric_curve, 0, end_value)
    metric_curve[-1] = end_value
    full_curve = np.concatenate([np.array(initial_data[metric_name][:start_epoch]), metric_curve])
    return full_curve

# Generate training metrics
data = {'epoch': range(1, epochs + 1)}
for metric_name in initial_data.keys():
    start_value = initial_data[metric_name][-1]
    end_value = final_data[metric_name]
    if metric_name == 'metrics/precision(B)':
        data[metric_name] = generate_metric_curve(start_value, end_value, epochs, metric_name, initial_data, noise_scale=0.02, growth_rate=0.03)
    elif metric_name == 'metrics/recall(B)':
        data[metric_name] = generate_metric_curve(start_value, end_value, epochs, metric_name, initial_data, noise_scale=0.015, growth_rate=0.035)
    elif metric_name == 'metrics/mAP50(B)':
        data[metric_name] = generate_metric_curve(start_value, end_value, epochs, metric_name, initial_data, noise_scale=0.01, growth_rate=0.04)
    elif metric_name == 'metrics/mAP50-95(B)':
        data[metric_name] = generate_metric_curve(start_value, end_value, epochs, metric_name, initial_data, noise_scale=0.01, growth_rate=0.025)

# Create DataFrame for training metrics
df = pd.DataFrame(data)
df.to_csv('yolo_metrics_training_data.csv', index=False)

# Updated function to generate a realistic PR curve
def generate_pr_curve(final_precision, final_recall, num_points=50):
    # Generate irregular recall points with clustering to mimic real thresholds
    recall = np.sort(np.concatenate([
        np.random.uniform(0, final_recall * 0.3, num_points // 3),  # Cluster early
        np.random.uniform(final_recall * 0.3, final_recall * 0.7, num_points // 3),  # Middle
        np.random.uniform(final_recall * 0.7, final_recall, num_points - 2 * (num_points // 3))  # End
    ]))
    recall = np.clip(recall, 0, final_recall)
    recall = np.concatenate([[0.0], recall, [final_recall]])  # Start at 0, end at target
    
    # Simulate precision with step-wise drops and noise
    precision = np.ones(len(recall))
    for i in range(1, len(recall)):
        drop = np.random.uniform(0.01, 0.05) * (1 - recall[i])  # Larger drops early
        noise = np.random.uniform(-0.02, 0.02)  # Add variability
        precision[i] = precision[i-1] - drop + noise
        precision[i] = min(precision[i-1], max(0, precision[i]))  # Enforce monotonicity
    
    # Scale precision to match final value
    precision = precision - (precision[-1] - final_precision)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # Ensure monotonicity
    precision = np.clip(precision, 0, 1)
    precision[0] = 1.0  # Start at 1
    precision[-1] = final_precision  # End at target
    
    return precision, recall

# Generate PR curve data
precision, recall = generate_pr_curve(final_data['metrics/precision(B)'], final_data['metrics/recall(B)'])

# Plot Precision-Recall Curve with step-like variations
plt.figure(figsize=(8, 6))
plt.step(recall, precision, where='post', label=f'PR Curve (AP={final_data["metrics/mAP50(B)"]:.3f})', color='b')
plt.title('Precision-Recall Curve (Final Model)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend()
plt.savefig('pr_curve.png')
plt.close()

# Plot mAP over epochs
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], 'g-', label='mAP50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'r-', label='mAP50-95')
plt.title('mAP Over Training Epochs')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.grid(True)
plt.legend()
plt.savefig('map_over_epochs.png')
plt.close()

print("Training metrics saved as 'yolo_metrics_training_data.csv'.")
print("Realistic PR curve saved as 'pr_curve.png'.")
print("mAP plot saved as 'map_over_epochs.png'.")