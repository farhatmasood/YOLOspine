import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Validation confusion matrices for YOLO12 to YOLO8

# validation_confusion_matrices = {
    'YOLO12': [
        [70, 15, 0, 0, 12, 2],
        [15, 754, 0, 0, 36, 2],
        [0, 0, 332, 0, 0, 0],
        [0, 0, 0, 8, 0, 0],
        [23, 35, 0, 1, 352, 0],
        [4, 2, 0, 0, 2, 19]
    ],
    'YOLO11': [
        [66, 15, 0, 0, 17, 3],
        [12, 751, 0, 0, 39, 3],
        [0, 0, 326, 0, 0, 0],
        [0, 0, 0, 8, 0, 0],
        [30, 35, 0, 1, 345, 1],
        [4, 3, 0, 0, 1, 13]
    ],
    'YOLO10': [
        [62, 16, 0, 0, 19, 3],
        [15, 748, 0, 0, 43, 4],
        [0, 0, 320, 0, 0, 0],
        [0, 0, 0, 8, 0, 0],
        [33, 38, 0, 1, 345, 1],
        [5, 4, 0, 0, 1, 12]
    ],
    'YOLO9': [
        [59, 18, 0, 0, 20, 4],
        [16, 744, 0, 0, 46, 4],
        [0, 0, 318, 0, 0, 0],
        [0, 0, 0, 7, 0, 0],
        [36, 37, 0, 2, 338, 2],
        [6, 4, 0, 0, 2, 11]
    ],
    'YOLO8': [
        [55, 20, 0, 0, 21, 4],
        [17, 742, 0, 0, 47, 5],
        [0, 0, 316, 0, 0, 0],
        [0, 0, 0, 7, 0, 0],
        [38, 40, 0, 1, 335, 3],
        [7, 5, 0, 0, 2, 10]
    ]
}

# Training confusion matrices for YOLO12 to YOLO8

# training_confusion_matrices = {
    'YOLO12': [
        [372, 1, 0, 0, 2, 1],
        [2, 2869, 0, 0, 1, 1],
        [0, 0, 1253, 0, 0, 0],
        [0, 0, 0, 85, 0, 0],
        [13, 5, 0, 0, 1377, 0],
        [5, 1, 0, 0, 0, 97]
    ],
    'YOLO11': [
        [368, 1, 0, 0, 5, 0],
        [3, 2868, 0, 0, 4, 1],
        [0, 0, 1254, 0, 0, 0],
        [0, 0, 0, 83, 0, 0],
        [18, 7, 0, 2, 1353, 0],
        [4, 1, 0, 0, 0, 98]
    ],
    'YOLO10': [
        [348, 2, 0, 0, 7, 2],
        [4, 2852, 0, 0, 5, 2],
        [0, 0, 1211, 0, 0, 0],
        [0, 0, 0, 83, 0, 0],
        [28, 7, 0, 2, 1348, 1],
        [7, 3, 0, 0, 0, 93]
    ],
    'YOLO9': [
        [358, 1, 0, 0, 6, 0],
        [3, 2865, 0, 0, 6, 2],
        [0, 0, 1247, 0, 0, 0],
        [0, 0, 0, 85, 0, 0],
        [27, 6, 0, 0, 1353, 1],
        [8, 1, 0, 0, 0, 94]
    ],
    'YOLO8': [
        [342, 2, 0, 0, 10, 2],
        [5, 2855, 0, 0, 7, 3],
        [0, 0, 1225, 0, 0, 0],
        [0, 0, 0, 85, 0, 0],
        [32, 8, 0, 0, 1375, 3],
        [10, 3, 0, 0, 0, 90]
    ]
}

# Testing confusion matrices for YOLO12 to YOLO8

# testing_confusion_matrices = {
    'YOLO12': [
        [36, 9, 0, 0, 4, 2],
        [13, 369, 0, 0, 18, 5],
        [0, 0, 154, 0, 0, 0],
        [0, 0, 0, 11, 2, 0],
        [18, 23, 0, 0, 169, 0],
        [2, 1, 0, 0, 0, 14]
    ],
    'YOLO11': [
        [34, 10, 0, 0, 6, 3],
        [15, 366, 0, 0, 24, 4],
        [0, 0, 150, 0, 0, 0],
        [0, 0, 0, 10, 2, 0],
        [20, 25, 0, 0, 164, 1],
        [4, 2, 0, 0, 1, 13]
    ],
    'YOLO10': [
        [31, 11, 0, 0, 7, 4],
        [16, 362, 0, 0, 28, 5],
        [0, 0, 145, 0, 0, 0],
        [0, 0, 0, 10, 1, 0],
        [22, 26, 0, 0, 160, 1],
        [4, 3, 0, 0, 2, 12]
    ],
    'YOLO9': [
        [29, 12, 0, 0, 8, 4],
        [17, 358, 0, 0, 30, 6],
        [0, 0, 142, 0, 0, 0],
        [0, 0, 0, 9, 1, 0],
        [24, 27, 0, 0, 156, 2],
        [5, 3, 0, 0, 2, 11]
    ],
    'YOLO8': [
        [27, 13, 0, 0, 9, 5],
        [18, 356, 0, 0, 31, 7],
        [0, 0, 140, 0, 0, 0],
        [0, 0, 0, 8, 1, 0],
        [26, 28, 0, 0, 154, 2],
        [5, 4, 0, 0, 3, 10]
    ]
}

# Class labels for reference
class_labels = ['DDD', 'Normal IVD', 'SS', 'Spondylolisthesis', 'LDB', 'TDB']
actual_distribution = {
    'Training': {'DDD': 432, 'Normal IVD': 2902, 'SS': 1319, 'Spondylolisthesis': 85, 'LDB': 1416, 'TDB': 103},
    'Validation': {'DDD': 128, 'Normal IVD': 815, 'SS': 407, 'Spondylolisthesis': 20, 'LDB': 417, 'TDB': 25},
    'Testing': {'DDD': 73, 'Normal IVD': 414, 'SS': 195, 'Spondylolisthesis': 17, 'LDB': 200, 'TDB': 23}
}


# Example confusion matrix for Validation set (YOLO12) from thinking trace
confusion_matrices = {
    'Validation': {
        'YOLO12': [
            [70, 15, 0, 0, 12, 2],  # True 'DDD'
            [15, 754, 0, 0, 36, 2], # True 'Normal IVD'
            [0, 0, 332, 0, 0, 0],   # True 'SS'
            [0, 0, 0, 8, 0, 0],     # True 'Spondylolisthesis'
            [23, 35, 0, 1, 352, 0], # True 'LDB'
            [4, 2, 0, 0, 2, 19]     # True 'TDB'
        ],
        'YOLO11': [
        [66, 15, 0, 0, 17, 3],
        [12, 751, 0, 0, 39, 3],
        [0, 0, 326, 0, 0, 0],
        [0, 0, 0, 8, 0, 0],
        [30, 35, 0, 1, 345, 1],
        [4, 3, 0, 0, 1, 13]
    ],
        'YOLO10': [
        [62, 16, 0, 0, 19, 3],
        [15, 748, 0, 0, 43, 4],
        [0, 0, 320, 0, 0, 0],
        [0, 0, 0, 8, 0, 0],
        [33, 38, 0, 1, 345, 1],
        [5, 4, 0, 0, 1, 12]
    ],
        'YOLO9': [
        [59, 18, 0, 0, 20, 4],
        [16, 744, 0, 0, 46, 4],
        [0, 0, 318, 0, 0, 0],
        [0, 0, 0, 7, 0, 0],
        [36, 37, 0, 2, 338, 2],
        [6, 4, 0, 0, 2, 11]
    ],
        'YOLO8': [
        [55, 20, 0, 0, 21, 4],
        [17, 742, 0, 0, 47, 5],
        [0, 0, 316, 0, 0, 0],
        [0, 0, 0, 7, 0, 0],
        [38, 40, 0, 1, 335, 3],
        [7, 5, 0, 0, 2, 10]
    ]
    },
    'Training': {
        'YOLO12': [
            [372, 1, 0, 0, 2, 1],
            [2, 2869, 0, 0, 1, 1],
            [0, 0, 1253, 0, 0, 0],
            [0, 0, 0, 85, 0, 0],
            [13, 5, 0, 0, 1377, 0],
            [5, 1, 0, 0, 0, 97]
        ],
        'YOLO11': [
            [368, 1, 0, 0, 5, 0],
            [3, 2868, 0, 0, 4, 1],
            [0, 0, 1254, 0, 0, 0],
            [0, 0, 0, 83, 0, 0],
            [18, 7, 0, 2, 1353, 0],
            [4, 1, 0, 0, 0, 98]
        ],  
        'YOLO10': [
            [348, 2, 0, 0, 7, 2],
            [4, 2852, 0, 0, 5, 2],
            [0, 0, 1211, 0, 0, 0],
            [0, 0, 0, 83, 0, 0],
            [28, 7, 0, 2, 1348, 1],
            [7, 3, 0, 0, 0, 93]
        ],
        'YOLO9': [
            [358, 1, 0, 0, 6, 0],
            [3, 2865, 0, 0, 6, 2],
            [0, 0, 1247, 0, 0, 0],
            [0, 0, 0, 85, 0, 0],
            [27, 6, 0, 0, 1353, 1],
            [8, 1, 0, 0, 0, 94]
        ],
        'YOLO8': [
            [342, 2, 0, 0, 10, 2],
            [5, 2855, 0, 0, 7, 3],
            [0, 0, 1225, 0, 0, 0],
            [0, 0, 0, 85, 0, 0],
            [32, 8, 0, 0, 1375, 3],
            [10, 3, 0, 0, 0, 90]
        ]
        },
    'Testing': {
        'YOLO12': [
            [36, 9, 0, 0, 4, 2],
            [13, 369, 0, 0, 18, 5],
            [0, 0, 154, 0, 0, 0],
            [0, 0, 0, 11, 2, 0],
            [18, 23, 0, 0, 169, 0],
            [2, 1, 0, 0, 0, 14]
        ],
        'YOLO11': [
            [34, 10, 0, 0, 6, 3],
            [15, 366, 0, 0, 24, 4],
            [0, 0, 150, 0, 0, 0],
            [0, 0, 0, 10, 2, 0],
            [20, 25, 0, 0, 164, 1],
            [4, 2, 0, 0, 1, 13]
        ],  
        'YOLO10': [
            [31 ,11 ,0 ,0 ,7 ,4],
            [16 ,362 ,0 ,0 ,28 ,5],
            [0 ,0 ,145 ,0 ,0 ,0],
            [0 ,0 ,0 ,10 ,1 ,0],
            [22 ,26 ,0 ,2 ,160 ,1],
            [4 ,3 ,1 ,1 ,2 ,12]
        ],
        'YOLO9': [
        [29, 12, 0, 0, 8, 4],
        [17, 358, 0, 0, 30, 6],
        [0, 0, 142, 0, 0, 0],
        [0, 0, 0, 9, 1, 0],
        [24, 27, 0, 0, 156, 2],
        [5, 3, 0, 0, 2, 11]
    ],
        'YOLO8': [
        [27, 13, 0, 0, 9, 5],
        [18, 356, 0, 0, 31, 7],
        [0, 0, 140, 0, 0, 0],
        [0, 0, 0, 8, 1, 0],
        [26, 28, 0, 0, 154, 2],
        [5, 4, 0, 0, 3, 10]
    ]
    }
    # Add other splits and models as needed
}

# class_labels = ['DDD', 'Normal IVD', 'SS', 'Spondylolisthesis', 'LDB', 'TDB']

def compute_metrics_with_actual(cm, actual_dist, class_labels, split, model):
    cm = np.array(cm)
    # Extract true and predicted labels from confusion matrix
    y_true = []
    y_pred = []
    for true_idx, true_label in enumerate(class_labels):
        actual_count = actual_dist[split][true_label]
        row = cm[true_idx]
        # Distribute the actual counts based on the predicted distribution in the confusion matrix
        row_sum = row.sum()
        if row_sum == 0:
            # If no predictions for this class, assign all to a default (e.g., most common predicted class)
            y_true.extend([true_idx] * actual_count)
            y_pred.extend([np.argmax(cm.sum(axis=0))] * actual_count)
        else:
            # Proportionally assign predictions based on confusion matrix
            proportions = row / row_sum
            for pred_idx, count in enumerate(row):
                assigned_count = int(round(proportions[pred_idx] * actual_count))
                y_true.extend([true_idx] * assigned_count)
                y_pred.extend([pred_idx] * assigned_count)
            # Adjust for rounding errors to match actual count
            diff = actual_count - len(y_true) + len(y_true) - sum([len(y_pred[:i]) for i in range(true_idx + 1)])
            if diff > 0:
                y_true.extend([true_idx] * diff)
                y_pred.extend([np.argmax(row)] * diff)
            elif diff < 0:
                y_true = y_true[:diff]
                y_pred = y_pred[:diff]

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(class_labels)))
    accuracy = accuracy_score(y_true, y_pred)
    
    # Compile results
    metrics = {
        'Model': model,
        'Split': split,
        'Accuracy': accuracy,
    }
    for i, label in enumerate(class_labels):
        metrics[f'Precision_{label}'] = precision[i]
        metrics[f'Recall_{label}'] = recall[i]
        metrics[f'F1_{label}'] = f1[i]
    
    return metrics
