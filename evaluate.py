import torch
import os
import logging
from Model.architecture import YOLOspine
from Model.dataset import create_dataloaders
from Model.loss import assign_targets, compute_stage1_loss, compute_stage2_loss, compute_grade_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, device, strides=[8, 16, 32]):
    """
    Evaluate model performance on test set
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_stage1_loss = 0.0
    total_stage2_loss = 0.0
    total_grade_loss = 0.0
    
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['targets']
            
            stage1_outputs, stage2_outputs = model(images)
            
            loss_stage1 = compute_stage1_loss(
                stage1_outputs,
                targets,
                device,
                strides=strides
            )
            
            pred_boxes = []
            for output, stride in zip(stage1_outputs, strides):
                reg_pred = output[:, 3:].permute(0, 2, 3, 1)
                B, H, W, _ = reg_pred.shape
                
                batch_idx_tensor = torch.arange(B, device=device).view(-1, 1).expand(-1, H*W).flatten()
                x, y, w_box, h_box, obj = reg_pred.reshape(-1, 5).split(1, dim=1)
                
                x1 = x - w_box / 2
                y1 = y - h_box / 2
                x2 = x + w_box / 2
                y2 = y + h_box / 2
                
                pred_boxes.append(torch.cat([
                    batch_idx_tensor.view(-1, 1),
                    x1, y1, x2, y2
                ], dim=1))
            
            pred_boxes = torch.cat(pred_boxes, dim=0) if pred_boxes else torch.empty((0, 5), device=device)
            assignments = assign_targets(pred_boxes, targets['boxes'].to(device))
            
            loss_stage2 = compute_stage2_loss(
                stage2_outputs,
                targets,
                device,
                assignments=assignments
            )
            
            loss_grade = compute_grade_loss(
                stage2_outputs,
                targets,
                device,
                assignments=assignments
            )
            
            loss = loss_stage1 + loss_stage2 + loss_grade
            
            total_loss += loss.item()
            total_stage1_loss += loss_stage1.item()
            total_stage2_loss += loss_stage2.item()
            total_grade_loss += loss_grade.item()
            num_batches += 1
    
    metrics = {
        'total_loss': total_loss / num_batches,
        'stage1_loss': total_stage1_loss / num_batches,
        'stage2_loss': total_stage2_loss / num_batches,
        'grade_loss': total_grade_loss / num_batches
    }
    
    return metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    data_dir = r'D:\2.3 Code_s\YOLOspine-2May25'
    grade_csv = r'D:\2.3 Code_s\YOLOspine-2May25\PfirrmannGrade.csv'
    checkpoint_path = r'D:\2.3 Code_s\YOLOspine-2May25\checkpoints\best.pth'
    
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir,
        grade_csv,
        batch_size=8,
        num_workers=0
    )
    
    logger.info("Initializing model...")
    model = YOLOspine(num_classes=6, num_grades=5)
    model = model.to(device)
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    else:
        logger.warning("No checkpoint found. Evaluating untrained model.")
    
    logger.info("Evaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, device)
    
    logger.info("\n=== Validation Metrics ===")
    logger.info(f"Total Loss: {val_metrics['total_loss']:.4f}")
    logger.info(f"Stage 1 Loss: {val_metrics['stage1_loss']:.4f}")
    logger.info(f"Stage 2 Loss: {val_metrics['stage2_loss']:.4f}")
    logger.info(f"Grade Loss: {val_metrics['grade_loss']:.4f}")
    
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    logger.info("\n=== Test Metrics ===")
    logger.info(f"Total Loss: {test_metrics['total_loss']:.4f}")
    logger.info(f"Stage 1 Loss: {test_metrics['stage1_loss']:.4f}")
    logger.info(f"Stage 2 Loss: {test_metrics['stage2_loss']:.4f}")
    logger.info(f"Grade Loss: {test_metrics['grade_loss']:.4f}")


if __name__ == '__main__':
    main()
