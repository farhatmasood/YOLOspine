import os
from ultralytics import YOLO

def summarize_model(model_path):
    try:
        # Load the model using YOLO's own loader
        model = YOLO(model_path)
        
        # Get model info
        model_info = str(model.model)
        
        # Calculate parameters
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.model.parameters() if not p.requires_grad)
        total_params = trainable_params + non_trainable_params
        
        # Create summary text
        model_summary = f"Summary for model: {os.path.basename(model_path)}\n\n"
        model_summary += f"{model_info}\n\n"
        model_summary += f"Trainable Parameters: {trainable_params:,}\n"
        model_summary += f"Non-Trainable Parameters: {non_trainable_params:,}\n"
        model_summary += f"Total Parameters: {total_params:,}\n"
        
        # Print to console
        print(model_summary)
        
        # Save to text file
        output_file = os.path.splitext(model_path)[0] + "_summary.txt"
        with open(output_file, "w") as f:
            f.write(model_summary)
        
        print(f"\nModel summary saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing model: {str(e)}")

def main():
    # Specify the model file path
    model_path = r"D:\2.3 Code_s\RF-Python - Copy-27Feb\Sagittal_T2_515_384x384\Split_Dataset\Model Results YOLOv8m to YOLOv12m\Models-PreTrained - YOLO\yolo12m.pt"
    
    if os.path.isfile(model_path):
        summarize_model(model_path)
    else:
        print(f"Model file not found: {model_path}")

if __name__ == "__main__":
    main()
