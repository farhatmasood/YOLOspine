import os
import numpy as np

class_names = {
    0: 'DDD',
    1: 'Normal_IVD',
    2: 'SS',
    3: 'Spondylolisthesis',
    4: 'LDB',
    5: 'TDB'
}

train_dir = 'd:\\2.3 Code_s\\YOLOspine-2May25\\train\\images'
label_dir = 'd:\\2.3 Code_s\\YOLOspine-2May25\\train\\labels'

test_files = [
    '0001_T2_TSE_SAG_384_0002_0002_008_R.png',
    '0002_T2_TSE_SAG_384_0002_0003_008_R.png'
]

print("="*80)
print("IVD ASSIGNMENT VERIFICATION")
print("="*80)
print(f"\nClass 1 (Normal_IVD) should be the ONLY class receiving Pfirrmann grades")
print(f"Pfirrmann grading measures disc degeneration (D3=L3-L4, D4=L4-L5, D5=L5-S1)\n")

for img_file in test_files:
    label_path = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')
    
    if not os.path.exists(label_path):
        print(f"Label not found: {label_path}")
        continue
    
    print(f"\nFile: {img_file}")
    
    boxes = []
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                c, x, y, w, h = map(float, parts[:5])
                boxes.append([x, y, w, h])
                labels.append(int(c))
    
    boxes = np.array(boxes, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    ivd_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
    
    print(f"  Total boxes: {len(boxes)}")
    print(f"  IVD boxes (class 1): {len(ivd_indices)}")
    
    if len(ivd_indices) > 0:
        ivd_boxes = boxes[ivd_indices]
        ivd_y_centers = ivd_boxes[:, 1]
        sorted_ivd_indices = np.argsort(ivd_y_centers)
        
        print(f"  IVD boxes sorted by Y-coordinate (top to bottom):")
        for rank, sorted_idx in enumerate(sorted_ivd_indices):
            orig_idx = ivd_indices[sorted_idx]
            y_pos = boxes[orig_idx][1]
            
            disc_level = None
            if len(ivd_indices) >= 3:
                if rank == len(ivd_indices) - 3:
                    disc_level = "D3 (L3-L4)"
                elif rank == len(ivd_indices) - 2:
                    disc_level = "D4 (L4-L5)"
                elif rank == len(ivd_indices) - 1:
                    disc_level = "D5 (L5-S1)"
            elif len(ivd_indices) == 2:
                if rank == len(ivd_indices) - 2:
                    disc_level = "D4 (L4-L5)"
                elif rank == len(ivd_indices) - 1:
                    disc_level = "D5 (L5-S1)"
            elif len(ivd_indices) == 1:
                disc_level = "D5 (L5-S1)"
            
            assignment = f" → {disc_level}" if disc_level else ""
            print(f"    IVD box {rank}: Y={y_pos:.3f}{assignment}")
    
    print(f"\n  All boxes (for reference):")
    for i, (box, lbl) in enumerate(zip(boxes, labels)):
        print(f"    Box {i}: Class={lbl} ({class_names[lbl]}), Y={box[1]:.3f}")

print("\n" + "="*80)
print("EXPECTED BEHAVIOR:")
print("  - Only class 1 (Normal_IVD) boxes should receive Pfirrmann grades")
print("  - Bottom 3 IVD boxes → D5, D4, D3 (based on Y-coordinate)")
print("  - Other classes (DDD, SS, Spondylolisthesis, LDB, TDB) should NOT get grades")
print("="*80)
