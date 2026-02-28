import sys
sys.path.insert(0, r'D:\2.3 Code_s\YOLOspine-2May25')

from Model.dataset import SpinalMRIDataset, load_pfirrmann_grades
import os

grade_csv = r'D:\2.3 Code_s\YOLOspine-2May25\PfirrmannGrade.csv'
grade_map = load_pfirrmann_grades(grade_csv)

test_filenames = [
    '0001_T2_TSE_SAG_384_0002_0002_008_R.png',
    '0580_T2_TSE_SAG_384_0002_0575_008_R.png',
    'T1_0001_S8 .png',
    'T1_0569_S8 .png'
]

dataset = SpinalMRIDataset(
    r'D:\2.3 Code_s\YOLOspine-2May25\train\images',
    r'D:\2.3 Code_s\YOLOspine-2May25\train\labels',
    class_names={0: 'Normal_IVD', 1: 'LDB', 2: 'SS', 3: 'DDD', 4: 'TDB', 5: 'Spondylolisthesis'},
    grade_map=grade_map
)

print("="*80)
print("PATIENT ID EXTRACTION VALIDATION")
print("="*80)

for filename in test_filenames:
    pid = dataset._extract_patient_id(filename)
    has_grades = pid in grade_map
    grades_info = f"D3={grade_map[pid][0]+1}, D4={grade_map[pid][1]+1}, D5={grade_map[pid][2]+1}" if has_grades else "NOT FOUND"
    
    print(f"\nFilename: {filename}")
    print(f"  Extracted Patient ID: {pid}")
    print(f"  In Grade CSV: {has_grades}")
    print(f"  Pfirrmann Grades: {grades_info}")

print("\n" + "="*80)
print("DISC LEVEL MAPPING (Anatomical Sorting)")
print("="*80)
print("Expected mapping based on Y-coordinate sorting:")
print("  - Bottom-most box (highest Y) → D5 (L5-S1)")
print("  - Middle box → D4 (L4-L5)")
print("  - Top-most box (lowest Y) → D3 (L3-L4)")
print("="*80)

img_files = [f for f in os.listdir(r'D:\2.3 Code_s\YOLOspine-2May25\train\images') if f.endswith('.png')]
sample_size = min(50, len(img_files))

matched = 0
unmatched = 0
format_breakdown = {'T1': 0, 'T2': 0, 'Other': 0}

for img_file in img_files[:sample_size]:
    pid = dataset._extract_patient_id(img_file)
    
    if img_file.startswith('T1_'):
        format_breakdown['T1'] += 1
    elif '_T2_TSE_SAG_' in img_file:
        format_breakdown['T2'] += 1
    else:
        format_breakdown['Other'] += 1
    
    if pid in grade_map:
        matched += 1
    else:
        unmatched += 1

print(f"\nSAMPLE ANALYSIS (First {sample_size} images):")
print(f"  Matched with Pfirrmann CSV: {matched}/{sample_size} ({matched/sample_size*100:.1f}%)")
print(f"  Unmatched: {unmatched}/{sample_size}")
print(f"\nFilename Format Distribution:")
print(f"  T1 format: {format_breakdown['T1']}")
print(f"  T2 format: {format_breakdown['T2']}")
print(f"  Other: {format_breakdown['Other']}")

print("\n" + "="*80)
print("CHECKING ACTUAL LABEL FILES")
print("="*80)

import numpy as np

for test_file in ['T1_0001_S8 .png', '0001_T2_TSE_SAG_384_0002_0002_008_R.png']:
    label_file = test_file.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(r'D:\2.3 Code_s\YOLOspine-2May25\train\labels', label_file)
    
    if os.path.exists(label_path):
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c, x, y, w, h = map(float, parts[:5])
                    boxes.append({'class': int(c), 'y': y, 'w': w, 'h': h})
        
        pid = dataset._extract_patient_id(test_file)
        
        print(f"\nFile: {test_file}")
        print(f"  Patient ID: {pid}")
        
        if pid in grade_map:
            grades = grade_map[pid]
            print(f"  Pfirrmann Grades: D3={grades[0]+1}, D4={grades[1]+1}, D5={grades[2]+1}")
            
            if len(boxes) > 0:
                boxes_sorted = sorted(boxes, key=lambda b: b['y'])
                print(f"  Number of boxes: {len(boxes)}")
                print(f"  Boxes sorted by Y-coordinate (top to bottom):")
                
                for i, box in enumerate(boxes_sorted):
                    grade_assignment = "N/A"
                    if len(boxes_sorted) >= 3:
                        if i == len(boxes_sorted) - 3:
                            grade_assignment = f"D3 (Grade {grades[0]+1})"
                        elif i == len(boxes_sorted) - 2:
                            grade_assignment = f"D4 (Grade {grades[1]+1})"
                        elif i == len(boxes_sorted) - 1:
                            grade_assignment = f"D5 (Grade {grades[2]+1})"
                    
                    print(f"    Box {i}: Class={box['class']}, Y={box['y']:.3f} → {grade_assignment}")
            else:
                print(f"  No boxes found in label file")
        else:
            print(f"  Patient ID not found in Pfirrmann CSV")
    else:
        print(f"\nFile: {test_file}")
        print(f"  Label file NOT FOUND: {label_path}")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
