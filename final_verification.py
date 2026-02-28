import sys
sys.path.append('d:\\2.3 Code_s\\YOLOspine-2May25')

from Model.dataset import SpinalMRIDataset, load_pfirrmann_grades
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class_names = {
    0: 'DDD',
    1: 'Normal_IVD',
    2: 'SS',
    3: 'Spondylolisthesis',
    4: 'LDB',
    5: 'TDB'
}

print("="*80)
print("FINAL PRE-TRAINING VERIFICATION")
print("="*80)

grade_map = load_pfirrmann_grades('d:\\2.3 Code_s\\YOLOspine-2May25\\PfirrmannGrade.csv')
print(f"\n[OK] Loaded Pfirrmann grades for {len(grade_map)} patients")

transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

dataset = SpinalMRIDataset(
    img_dir='d:\\2.3 Code_s\\YOLOspine-2May25\\train\\images',
    label_dir='d:\\2.3 Code_s\\YOLOspine-2May25\\train\\labels',
    class_names=class_names,
    transform=transform,
    grade_map=grade_map
)

print(f"[OK] Created dataset with {len(dataset)} images\n")

print("="*80)
print("TESTING SAMPLE IMAGES")
print("="*80)

num_samples = 10
matched_count = 0
ivd_grade_count = 0
non_ivd_grade_count = 0

for i in range(min(num_samples, len(dataset))):
    sample = dataset[i]
    img_file = dataset.img_files[i]
    patient_id = dataset._extract_patient_id(img_file)
    
    grades = sample['grades']
    labels = sample['labels']
    
    has_pfirrmann = patient_id in grade_map
    if has_pfirrmann:
        matched_count += 1
    
    print(f"\nImage {i+1}: {img_file}")
    print(f"  Patient ID: {patient_id} | In CSV: {has_pfirrmann}")
    
    if has_pfirrmann:
        print(f"  Expected grades: D3={grade_map[patient_id][0]+1}, D4={grade_map[patient_id][1]+1}, D5={grade_map[patient_id][2]+1}")
    
    print(f"  Total boxes: {len(labels)}")
    
    ivd_count = sum(1 for lbl in labels if lbl == 1)
    print(f"  IVD boxes (class 1): {ivd_count}")
    
    for j, (lbl, grade) in enumerate(zip(labels, grades)):
        class_name = class_names[lbl]
        grade_str = f"Grade {grade+1}" if grade >= 0 else "N/A"
        
        if grade >= 0:
            if lbl == 1:
                ivd_grade_count += 1
                status = "[OK]"
            else:
                non_ivd_grade_count += 1
                status = "[ERROR]"
            print(f"    Box {j}: {class_name} -> {grade_str} {status}")

print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)
print(f"[OK] Patient ID Match Rate: {matched_count}/{num_samples} ({matched_count/num_samples*100:.1f}%)")
print(f"[OK] IVD boxes with grades: {ivd_grade_count}")
print(f"[CHECK] Non-IVD boxes with grades: {non_ivd_grade_count}")

if non_ivd_grade_count > 0:
    print("\n[ERROR] Non-IVD boxes should NOT receive Pfirrmann grades!")
    print("        Pfirrmann grading is only for intervertebral discs (class 1)")
else:
    print("\n[SUCCESS] Only IVD boxes receive Pfirrmann grades (correct)")

print("\n" + "="*80)
print("ARCHITECTURE SUMMARY")
print("="*80)
print("Stage 1 Classes (3): DDD, Normal_IVD, SS")
print("Stage 2 Classes (3): Spondylolisthesis, LDB, TDB")
print("Pfirrmann Grades (5): I, II, III, IV, V")
print("Grade Assignment: Only class 1 (Normal_IVD) boxes")
print("  - Bottom-most IVD -> D5 (L5-S1)")
print("  - Middle IVD -> D4 (L4-L5)")
print("  - Top-most IVD (of bottom 3) -> D3 (L3-L4)")
print("="*80)

if non_ivd_grade_count == 0 and matched_count > 0:
    print("\n*** ALL CHECKS PASSED - READY FOR TRAINING ***\n")
else:
    print("\n*** ISSUES FOUND - DO NOT TRAIN YET ***\n")
