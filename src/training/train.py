# src/training/train.py

import os 
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import grad_scaler, autocast_mode
from tqdm import tqdm

# import previous modules
from src.data.dataset import MultimodalTrailDataset
from src.models.trail_net import MultiModalNet

class DiceLoss(nn.Module):
    """
    Directly optimizes for mask overlap (IoU) to mitigate massive background class imbalance
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        trail_probs = probs[:, 1, :, :]

        intersection = (trail_probs * targets).sum(dim=(1,2))
        denominator = trail_probs.sum(dim=(1,2)) + targets.sum(dim=(1,2))

        # avoid divide by zeros
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()
    
def calculate_iou(preds, targets):
    """
    Computes the Intersection over Union for the trails
    """

    intersection = ((preds == 1) & (targets == 1)).float().sum()
    union = ((preds==1) | (targets==1)).float().sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()

def train_model():
    # define paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    naip_path = os.path.join(PROJECT_ROOT, "data/raw/mt_tamalpais_naip.tif")
    elev_path = os.path.join(PROJECT_ROOT, "data/raw/mt_tamalpais_elevation.tif")
    mask_path = os.path.join(PROJECT_ROOT, "data/masks/mt_tamalpais_mask.tif")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # define hyperparameters
    epochs = 15
    batch_size = 4
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Starting training on: {device}')

    # pipeline acceleration
    dataset = MultimodalTrailDataset(naip_path=naip_path, elev_path=elev_path, 
                                     mask_path=mask_path, tile_size=512, stride=256)
    
    # definte 80/20 split for train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset=dataset, lengths=[train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # define model, optimizer, & balanced loss engine
    model = MultiModalNet(num_classes=2).to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    # apply 1:10 penalization scaling on Cross Entropy so more attention on thin trails
    ce_weight = torch.tensor([1.0, 10.0]).to(device=device)
    criterion_ce = nn.CrossEntropyLoss(weight=ce_weight)
    criterion_dice = DiceLoss()

    # AMP Precision Scaler to save hardware memory footprint
    scaler = grad_scaler.GradScaler("cuda")

    # begin training loop
    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")
        for visual, elev, targets in loop:
            visual, elev, targets = visual.to(device), elev.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # mix precision forward pass
            with autocast_mode.autocast("cuda"):
                outputs = model(visual, elev)
                loss_ce = criterion_ce(outputs, targets)
                loss_dice = criterion_dice(outputs, targets)
                loss = loss_ce + loss_dice
                
            # scaled gradiants backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # save training metrics
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_iou += calculate_iou(preds, targets)
            
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)

        # validation steps
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for visual, elev, targets in val_loader:
                visual, elev, targets = visual.to(device), elev.to(device), targets.to(device)
                
                with autocast_mode.autocast(device_type="cuda"):
                    outputs = model(visual, elev)
                    loss_ce = criterion_ce(outputs, targets)
                    loss_dice = criterion_dice(outputs, targets)
                    loss = loss_ce + loss_dice
                    
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_iou += calculate_iou(preds, targets)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f"\nMetrics:\nTrain Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f} || Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # save model tracking snapshots
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_trail_model.pth"))
            print(f"New Best Validation IoU achieved! Checkpoint saved.")

if __name__ == "__main__":
    train_model()

