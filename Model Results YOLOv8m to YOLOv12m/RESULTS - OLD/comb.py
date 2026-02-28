import shutil
import os

def merge_folders(src1, src2, dst):
    os.makedirs(dst, exist_ok=True)
    for folder in [src1, src2]:
        for fname in os.listdir(folder):
            full_src = os.path.join(folder, fname)
            full_dst = os.path.join(dst, fname)
            shutil.copy(full_src, full_dst)

# Merge images and labels
# merge_folders("train/images", "train/images_aug", "train/images_combined")
# merge_folders("train/labels", "train/labels_aug", "train/labels_combined")
merge_folders("val/images", "val/images_aug", "val/images_combined")
merge_folders("val/labels", "val/labels_aug", "val/labels_combined")