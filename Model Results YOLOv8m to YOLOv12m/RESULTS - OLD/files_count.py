import os

def count_files_in_subfolders(project_dir, image_extensions=[".png"], label_extensions=[".txt"]):
    """
    Count the number of image and label files in each subdirectory of project_dir and get their names.
    """
    summary = {}
    for root, dirs, files in os.walk(project_dir):
        # Count files in the current directory
        image_files_in_dir = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        label_files_in_dir = [f for f in files if os.path.splitext(f)[1].lower() in label_extensions]
        summary[root] = {
            "images": len(image_files_in_dir),
            "labels": len(label_files_in_dir)
        }
    return summary

# Define the project directory
project_dir = r"D:\2.3 Code_s\RF-Python - Copy-27Feb\Sagittal_T2_515_384x384\Split_Dataset"

# Count files and get their names
summary = count_files_in_subfolders(project_dir)

# Print the summary
print("Summary (Number of files in each subdirectory):")
for dir_path, counts in summary.items():
    print(f"{dir_path:100s} : Images: {counts['images']}, Labels: {counts['labels']}")