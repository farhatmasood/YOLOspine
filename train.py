import torch
import torch.optim as optim
import os
import logging
import argparse
from datetime import datetime

from Model.architecture import YOLOspine
from Model.dataset import create_dataloaders
from Model.loss import (
    assign_targets,
    compute_stage1_loss,
    compute_stage2_loss,
    compute_grade_loss,
    save_checkpoint,
    load_checkpoint
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Training configuration"""
    IMG_SIZE = (384, 384)
    NUM_CLASSES = 6
    NUM_GRADES = 5
    STRIDES = [8, 16, 32]
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    LAMBDA_REG = 1.0
    LAMBDA_GRADE = 0.5
    IOU_THRESHOLD = 0.2
    OBJ_THRESHOLD = 0.3
    MIN_BOX_SIZE = 1e-4
    
    DATA_DIR = r'D:\2.3 Code_s\YOLOspine-2May25'
    GRADE_CSV_PATH = r'D:\2.3 Code_s\YOLOspine-2May25\PfirrmannGrade.csv'
    CHECKPOINT_DIR = r'D:\2.3 Code_s\YOLOspine-2May25\checkpoints'
    
    SAVE_INTERVAL = 5
    LOG_INTERVAL = 10


def train_one_epoch(model, dataloader, optimizer, device, config, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_stage1_loss = 0.0
    total_stage2_loss = 0.0
    total_grade_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        targets = batch['targets']
        
        optimizer.zero_grad()
        
        stage1_outputs, stage2_outputs = model(images)
        
        loss_stage1 = compute_stage1_loss(
            stage1_outputs,
            targets,
            device,
            strides=config.STRIDES,
            lambda_reg=config.LAMBDA_REG,
            min_box_size=config.MIN_BOX_SIZE
        )
        
        pred_boxes = []
        for output, stride in zip(stage1_outputs, config.STRIDES):
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
        assignments = assign_targets(pred_boxes, targets['boxes'].to(device), iou_threshold=config.IOU_THRESHOLD)
        
        loss_stage2 = compute_stage2_loss(
            stage2_outputs,
            targets,
            device,
            assignments=assignments,
            lambda_reg=config.LAMBDA_REG
        )
        
        loss_grade = compute_grade_loss(
            stage2_outputs,
            targets,
            device,
            assignments=assignments,
            lambda_grade=config.LAMBDA_GRADE
        )
        
        loss = loss_stage1 + loss_stage2 + loss_grade
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_stage1_loss += loss_stage1.item()
        total_stage2_loss += loss_stage2.item()
        total_grade_loss += loss_grade.item()
        
        if batch_idx % config.LOG_INTERVAL == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"(S1: {loss_stage1.item():.4f}, "
                f"S2: {loss_stage2.item():.4f}, "
                f"Grade: {loss_grade.item():.4f})"
            )
    
    avg_loss = total_loss / len(dataloader)
    avg_stage1 = total_stage1_loss / len(dataloader)
    avg_stage2 = total_stage2_loss / len(dataloader)
    avg_grade = total_grade_loss / len(dataloader)
    
    logger.info(
        f"Epoch {epoch} Training Summary - "
        f"Avg Loss: {avg_loss:.4f} "
        f"(S1: {avg_stage1:.4f}, S2: {avg_stage2:.4f}, Grade: {avg_grade:.4f})"
    )
    
    return avg_loss


def validate(model, dataloader, device, config, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_stage1_loss = 0.0
    total_stage2_loss = 0.0
    total_grade_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['targets']
            
            stage1_outputs, stage2_outputs = model(images)
            
            loss_stage1 = compute_stage1_loss(
                stage1_outputs,
                targets,
                device,
                strides=config.STRIDES,
                lambda_reg=config.LAMBDA_REG,
                min_box_size=config.MIN_BOX_SIZE
            )
            
            pred_boxes = []
            for output, stride in zip(stage1_outputs, config.STRIDES):
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
            assignments = assign_targets(pred_boxes, targets['boxes'].to(device), iou_threshold=config.IOU_THRESHOLD)
            
            loss_stage2 = compute_stage2_loss(
                stage2_outputs,
                targets,
                device,
                assignments=assignments,
                lambda_reg=config.LAMBDA_REG
            )
            
            loss_grade = compute_grade_loss(
                stage2_outputs,
                targets,
                device,
                assignments=assignments,
                lambda_grade=config.LAMBDA_GRADE
            )
            
            loss = loss_stage1 + loss_stage2 + loss_grade
            
            total_loss += loss.item()
            total_stage1_loss += loss_stage1.item()
            total_stage2_loss += loss_stage2.item()
            total_grade_loss += loss_grade.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_stage1 = total_stage1_loss / len(dataloader)
    avg_stage2 = total_stage2_loss / len(dataloader)
    avg_grade = total_grade_loss / len(dataloader)
    
    logger.info(
        f"Epoch {epoch} Validation - "
        f"Avg Loss: {avg_loss:.4f} "
        f"(S1: {avg_stage1:.4f}, S2: {avg_stage2:.4f}, Grade: {avg_grade:.4f})"
    )
    
    return avg_loss


def train(config):
    """Main training function"""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        config.DATA_DIR,
        config.GRADE_CSV_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=0
    )
    
    model = YOLOspine(num_classes=config.NUM_CLASSES, num_grades=config.NUM_GRADES)
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
    if os.path.exists(checkpoint_path):
        try:
            start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
            start_epoch += 1
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    
    logger.info(f"Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config, epoch)
        val_loss = validate(model, val_loader, device, config, epoch)
        
        scheduler.step(val_loss)
        
        if epoch % config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        latest_path = os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, latest_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.CHECKPOINT_DIR, 'best.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='YOLOspine Training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default=r'D:\2.3 Code_s\YOLOspine-2May25', help='Data directory')
    
    args = parser.parse_args()
    
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.DATA_DIR = args.data_dir
    
    train(config)


if __name__ == '__main__':
    main()
