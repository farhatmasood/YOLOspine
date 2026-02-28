# import os

# image_dir = "train/images_combined"
# label_dir = "train/labels_combined"

# for f in os.listdir(image_dir):
#     if f.endswith(".png"):
#         label_file = os.path.join(label_dir, f.replace(".png", ".txt"))
#         if not os.path.exists(label_file):
#             print("❌ MISSING:", label_file)
#         elif os.path.getsize(label_file) == 0:
#             print("⚠️ EMPTY :", label_file)

import os

image_dir = "train/images_combined"
label_dir = "train/labels_combined"

def is_valid_line(line):
    try:
        parts = line.strip().split()
        if len(parts) != 5:
            return False
        floats = list(map(float, parts))
        return all(0 <= x <= 1 for x in floats[1:])  # class_id can be int, coords in 0-1
    except:
        return False

for f in os.listdir(image_dir):
    if f.endswith(".png"):
        label_path = os.path.join(label_dir, f.replace(".png", ".txt"))
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
                if not lines:
                    print("⚠️ EMPTY :", label_path)
                else:
                    valid_lines = [line for line in lines if is_valid_line(line)]
                    if len(valid_lines) == 0:
                        print("❌ INVALID FORMAT:", label_path)
        else:
            print("❌ MISSING:", label_path)
