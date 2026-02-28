"""
YOLOspine Setup Verification Script

Checks that all required components are properly configured.
Run this before starting training to catch configuration issues early.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path


def check_directories(base_path):
    """Verify required directory structure exists"""
    print("=" * 60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    required_dirs = [
        'train/images',
        'train/labels',
        'val/images',
        'val/labels',
        'test/images',
        'test/labels',
        'Model',
        'checkpoints'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        exists = os.path.exists(full_path)
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_files(base_path):
    """Verify required files exist"""
    print("\n" + "=" * 60)
    print("CHECKING REQUIRED FILES")
    print("=" * 60)
    
    required_files = [
        'data.yaml',
        'PfirrmannGrade.csv',
        'train.py',
        'evaluate.py',
        'requirements.txt',
        'README.md',
        'Model/__init__.py',
        'Model/architecture.py',
        'Model/dataset.py',
        'Model/loss.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        exists = os.path.exists(full_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_dataset(base_path):
    """Verify dataset integrity"""
    print("\n" + "=" * 60)
    print("CHECKING DATASET")
    print("=" * 60)
    
    splits = ['train', 'val', 'test']
    dataset_stats = {}
    
    for split in splits:
        img_dir = os.path.join(base_path, split, 'images')
        label_dir = os.path.join(base_path, split, 'labels')
        
        if os.path.exists(img_dir) and os.path.exists(label_dir):
            num_images = len([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))])
            num_labels = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])
            
            dataset_stats[split] = {
                'images': num_images,
                'labels': num_labels,
                'matched': num_images == num_labels
            }
            
            status = "✓" if num_images == num_labels else "⚠"
            print(f"{status} {split}: {num_images} images, {num_labels} labels")
        else:
            print(f"✗ {split}: Directory not found")
            dataset_stats[split] = {'images': 0, 'labels': 0, 'matched': False}
    
    return all(stats['matched'] for stats in dataset_stats.values())


def check_pfirrmann_grades(base_path):
    """Verify Pfirrmann grade CSV"""
    print("\n" + "=" * 60)
    print("CHECKING PFIRRMANN GRADES")
    print("=" * 60)
    
    csv_path = os.path.join(base_path, 'PfirrmannGrade.csv')
    
    if not os.path.exists(csv_path):
        print("✗ PfirrmannGrade.csv not found")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        required_cols = ['Patient_ID', 'D3', 'D4', 'D5']
        
        has_cols = all(col in df.columns for col in required_cols)
        if not has_cols:
            print(f"✗ Missing columns. Expected: {required_cols}")
            return False
        
        num_patients = len(df)
        grade_range_valid = all(
            df[col].between(1, 5).all() for col in ['D3', 'D4', 'D5']
        )
        
        print(f"✓ {num_patients} patients with Pfirrmann grades")
        
        if not grade_range_valid:
            print("⚠ Warning: Some grades outside valid range [1-5]")
        
        print(f"  - D3 range: {df['D3'].min()} to {df['D3'].max()}")
        print(f"  - D4 range: {df['D4'].min()} to {df['D4'].max()}")
        print(f"  - D5 range: {df['D5'].min()} to {df['D5'].max()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return False


def check_dependencies():
    """Verify Python package dependencies"""
    print("\n" + "=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('yaml', 'PyYAML'),
        ('albumentations', 'Albumentations')
    ]
    
    all_installed = True
    for pkg_name, display_name in required_packages:
        try:
            __import__(pkg_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def check_cuda():
    """Check CUDA availability"""
    print("\n" + "=" * 60)
    print("CHECKING CUDA")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA available")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ CUDA not available - training will use CPU (slower)")
    
    return True


def main():
    base_path = r'D:\2.3 Code_s\YOLOspine-2May25'
    
    print("\n" + "=" * 60)
    print("YOLOspine Setup Verification")
    print("=" * 60)
    print(f"Base path: {base_path}\n")
    
    checks = {
        'Directories': check_directories(base_path),
        'Files': check_files(base_path),
        'Dataset': check_dataset(base_path),
        'Pfirrmann Grades': check_pfirrmann_grades(base_path),
        'Dependencies': check_dependencies(),
        'CUDA': check_cuda()
    }
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {check_name}")
    
    all_passed = all(checks.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("\nTo start training, run:")
        print("  python train.py --batch_size 8 --epochs 100")
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues before training")
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Missing directories: Create train/val/test folders")
        print("  - Dataset issues: Verify images and labels match")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
