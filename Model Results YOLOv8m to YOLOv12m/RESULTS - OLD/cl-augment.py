import os
import cv2
import random
import yaml
import numpy as np
import albumentations as A

# ---------------------------------------------
# Load class names from data.yaml
# ---------------------------------------------
def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

# ---------------------------------------------
# Read YOLO format labels (class x_center y_center width height)
# ---------------------------------------------
def read_yolo_labels(label_path):
    bboxes = []
    category_ids = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_c, y_c, w, h = map(float, parts)
                    bboxes.append([x_c, y_c, w, h])
                    category_ids.append(int(class_id))
    return bboxes, category_ids

# ---------------------------------------------
# Write YOLO format labels to .txt
# ---------------------------------------------
def write_yolo_labels(label_path, category_ids, bboxes):
    with open(label_path, 'w') as f:
        for cls, bbox in zip(category_ids, bboxes):
            x_c, y_c, w, h = bbox
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

# ---------------------------------------------
# Count number of images per class in dataset
# ---------------------------------------------
def count_classes(image_dir, label_dir):
    class_counts = {}
    image_class_map = {}
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.png'):
            label_file = img_file.replace('.png', '.txt')
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                _, category_ids = read_yolo_labels(label_path)
                image_class_map[img_file] = set(category_ids)
                for cls in set(category_ids):
                    class_counts[cls] = class_counts.get(cls, 0) + 1
    return class_counts, image_class_map

# Random Noise Patch
def random_noise_patch(image, **kwargs):
    h, w = image.shape[:2]
    max_h, max_w = 32, 32
    patch_h = random.randint(16, max_h)
    patch_w = random.randint(16, max_w)
    y = random.randint(0, h - patch_h)
    x = random.randint(0, w - patch_w)

    # Create random noise
    noise = np.random.randint(0, 256, (patch_h, patch_w), dtype=np.uint8)

    if image.ndim == 2:
        image[y:y+patch_h, x:x+patch_w] = noise
    else:
        for c in range(image.shape[2]):
            image[y:y+patch_h, x:x+patch_w, c] = noise
    return image  # ✅ return only the image (not a dict)

# ---------------------------------------------
# Augment image and filter invalid bounding boxes
# ---------------------------------------------
def augment_image_and_labels(img_path, label_path, out_img_dir, out_label_dir, transform, aug_index):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Unable to read {img_path}")
        return False
    bboxes, category_ids = read_yolo_labels(label_path)
    if not bboxes:
        return False

    pre_clipped_bboxes = []
    for bbox in bboxes:
        x_c, y_c, w, h = bbox
        x_c = max(0, min(1, x_c))
        y_c = max(0, min(1, y_c))
        w = min(w, 2 * (1 - x_c) if x_c > 0.5 else 2 * x_c)
        h = min(h, 2 * (1 - y_c) if y_c > 0.5 else 2 * y_c)
        if w > 0 and h > 0:
            pre_clipped_bboxes.append([x_c, y_c, w, h])
    if not pre_clipped_bboxes:
        return False

    base = os.path.splitext(os.path.basename(img_path))[0]
    try:
        augmented = transform(image=img, bboxes=pre_clipped_bboxes, category_ids=category_ids)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_category_ids = augmented['category_ids']

        filtered_bboxes = []
        filtered_category_ids = []
        for bbox, cls in zip(aug_bboxes, aug_category_ids):
            x_c, y_c, w, h = bbox
            x_c = max(0, min(1, x_c))
            y_c = max(0, min(1, y_c))
            w = min(w, 2 * (1 - x_c) if x_c > 0.5 else 2 * x_c)
            h = min(h, 2 * (1 - y_c) if y_c > 0.5 else 2 * y_c)
            if w > 0 and h > 0:
                filtered_bboxes.append([x_c, y_c, w, h])
                filtered_category_ids.append(cls)

        if filtered_bboxes:
            new_base = f"{base}_aug_{aug_index}"
            out_img_path = os.path.join(out_img_dir, new_base + ".png")
            out_label_path = os.path.join(out_label_dir, new_base + ".txt")
            cv2.imwrite(out_img_path, aug_img)
            write_yolo_labels(out_label_path, filtered_category_ids, filtered_bboxes)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error during augmentation for {img_path}: {e}")
        return False

# ---------------------------------------------
# Balanced class-wise augmentation
# ---------------------------------------------
def process_dataset(image_dir, label_dir, out_img_dir, out_label_dir, transform, class_names):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    class_counts, image_class_map = count_classes(image_dir, label_dir)
    print_class_counts(class_counts, "Initial Counts Before Augmentation", class_names)

    target_count = max(class_counts.values())
    aug_needed = {cls: max(0, target_count - count) for cls, count in class_counts.items()}
    print(f"\nAugmentations needed per class to reach {target_count}:")
    for cls in sorted(aug_needed.keys()):
        print(f"{class_names[cls]}: {aug_needed[cls]}")

    all_images = [f for f in os.listdir(image_dir) if f.endswith('.png') and os.path.exists(os.path.join(label_dir, f.replace('.png', '.txt')))]
    aug_index = 0

    for cls in sorted(aug_needed.keys()):
        needed = aug_needed[cls]
        if needed == 0:
            continue

        print(f"\n🔄 Augmenting class {cls} - {class_names[cls]} (Need {needed})")
        images_with_class = [img for img in all_images if cls in image_class_map.get(img, [])]

        if not images_with_class:
            print(f"⚠️ No images found for class {cls} - {class_names[cls]}")
            continue

        i = 0
        while class_counts[cls] < target_count and i < 10000:
            img_file = random.choice(images_with_class)
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.replace('.png', '.txt'))

            success = augment_image_and_labels(img_path, label_path, out_img_dir, out_label_dir, transform, aug_index)
            if success:
                aug_index += 1
                new_label_file = os.path.join(out_label_dir, f"{os.path.splitext(img_file)[0]}_aug_{aug_index - 1}.txt")
                _, new_classes = read_yolo_labels(new_label_file)
                for cat in new_classes:
                    if class_counts.get(cat, 0) < target_count:
                        class_counts[cat] = class_counts.get(cat, 0) + 1
            i += 1

    print_class_counts(class_counts, "Final Counts After Augmentation", class_names)

# ---------------------------------------------
# Class count printer
# ---------------------------------------------
def print_class_counts(counts, title, class_names):
    print(f"\n{title}:")
    print("-" * 40)
    total = sum(counts.values())
    for cls in sorted(counts.keys()):
        print(f"{class_names[cls]}: {counts[cls]}")
    print(f"Total class instances: {total}")
    print("-" * 40)

# ---------------------------------------------
# Main Execution
# ---------------------------------------------
base_path = r"D:\2.3 Code_s\RF-Python - Copy-27Feb\Sagittal_T2_515_384x384\Split_Dataset"
yaml_path = os.path.join(base_path, "data.yaml")

# Load class names
class_names = load_class_names(yaml_path)

# Define folders
train_img_dir = os.path.join(base_path, "train", "images")
val_img_dir = os.path.join(base_path, "val", "images")
train_label_dir = os.path.join(base_path, "train", "labels")
val_label_dir = os.path.join(base_path, "val", "labels")
train_out_img_dir = os.path.join(base_path, "train", "images_aug")
train_out_label_dir = os.path.join(base_path, "train", "labels_aug")
val_out_img_dir = os.path.join(base_path, "val", "images_aug")
val_out_label_dir = os.path.join(base_path, "val", "labels_aug")

# Augmentation Transform
transform = A.Compose([
    A.SomeOf([
        A.HorizontalFlip(p=1.0),
        A.Rotate(limit=20, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.GaussianBlur(blur_limit=3, p=1.0),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.ElasticTransform(alpha=1.0, sigma=50, p=1.0),
        A.Lambda(image=random_noise_patch, p=1.0),  # ✅ custom noise patch
    ], n=2, replace=False),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.3))


# Training set
print("Processing training dataset...")
process_dataset(train_img_dir, train_label_dir, train_out_img_dir, train_out_label_dir, transform, class_names)

# Validation set
print("\nProcessing validation dataset...")
process_dataset(val_img_dir, val_label_dir, val_out_img_dir, val_out_label_dir, transform, class_names)

print("\n✅ Augmentation completed successfully.")
