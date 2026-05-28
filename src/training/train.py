# src/training/train.py
import os
os.environ['GDAL_DATA'] = r'C:\Users\flyin\Miniconda3\envs\hike\Lib\site-packages\osgeo\data\gdal'
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import grad_scaler, autocast_mode
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import MultimodalTrailDataset
from src.models.trail_net import MultiModalNet
from src.utils.config_loader import load_region_config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

        # ensure targets are float, on the right device, and match shape
        targets = targets.float().to(device=logits.device)

        intersection = (trail_probs * targets).sum(dim=(1, 2))
        denominator = trail_probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))

        # avoid divide by zeros
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()


def calculate_iou(preds, targets):
    """
    Computes the Intersection over Union for the trails
    """

    intersection = ((preds == 1) & (targets == 1)).float().sum()
    union = ((preds == 1) | (targets == 1)).float().sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()


# Smoke Test
def run_smoke_test(model, train_loader, criterion_ce, criterion_dice, device):
    """
    Runs one batch through the multimodal training pipeline before full training 
    verify that dataset, model, and loss functions work before full training loop begins
    """

    print("\nRunning one-batch smoke test")

    # keep the model in training mode because this is testing the training pipeline
    model.train()

    # grab one batch from the training loader
    visual, elev, targets = next(iter(train_loader))

    # move all tensors to the same device as the model
    visual = visual.to(device)
    elev = elev.to(device)
    targets = targets.long().to(device)

    # print tensor shapes to verify the dataset and model match
    print("Visual shape:", visual.shape, "Expected: [B, 5, 512, 512]")
    print("Elevation shape:", elev.shape, "Expected: [B, 1, 512, 512]")
    print("Mask shape:", targets.shape, "Expected: [B, 512, 512]")
    

    with autocast_mode.autocast("cuda"):
        # forward pass 
        outputs = model(visual, elev)
        print("Output shape:", outputs.shape, "Expected: [B, 2, 512, 512]")

        # calculate the same losses used in training
        loss_ce = criterion_ce(outputs, targets)
        loss_dice = criterion_dice(outputs, targets)
        loss = loss_ce + loss_dice

    print("Smoke test CE loss:", loss_ce.item())
    print("Smoke test Dice loss:", loss_dice.item())
    print("Smoke test total loss:", loss.item())

    # backward pass check
    # confirms gradients can be computed.
    # no optimizer.step() is called, so weights are NOT updated.
    model.zero_grad()
    loss.backward()
    model.zero_grad()
    del outputs, loss, loss_ce, loss_dice, visual, elev, targets
    torch.cuda.empty_cache()

    print("Smoke test passed. Forward pass, loss, and backward pass all work.\n")


def train_model():
    # define paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    config = load_region_config(PROJECT_ROOT)

    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # define hyperparameters
    epochs = 15
    batch_size = 4
    accumulation_steps = 4
    learning_rate = 5e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Starting training on: {device}')

    # pipeline acceleration
    dataset = MultimodalTrailDataset(config=config)

    # define 80/20 split for train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[train_size, val_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # define model, optimizer, & balanced loss engine
    model = MultiModalNet(num_classes=2).to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # adaptive scheduler instead of blind cosine drops
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=2,
        factor=0.5,
        verbose=True
    )

    # apply 1:10 penalization scaling on Cross Entropy so more attention on thin trails
    ce_weight = torch.tensor([1.0, 10.0]).to(device=device)
    criterion_ce = nn.CrossEntropyLoss(weight=ce_weight)
    criterion_dice = DiceLoss()

    # -----------------------------
    # NHI ADDITION: Run Smoke Test
    # -----------------------------
    # This verifies that the dataset, model, and loss functions work
    # before the full training loop begins.
    run_smoke_test(
        model=model,
        train_loader=train_loader,
        criterion_ce=criterion_ce,
        criterion_dice=criterion_dice,
        device=device
    )

    # AMP Precision Scaler to save hardware memory footprint
    scaler = grad_scaler.GradScaler("cuda")

    # begin training loop
    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_iou = 0.0

        optimizer.zero_grad()
        optimization_steps = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{epochs}]")
        for batch_idx, (visual, elev, targets) in loop:
            visual, elev, targets = visual.to(device), elev.to(device), targets.to(device)

            # mix precision forward pass
            with autocast_mode.autocast("cuda"):
                outputs = model(visual, elev)
                loss_ce = criterion_ce(outputs, targets)
                loss_dice = criterion_dice(outputs, targets)

                # normalize loss to account for accumulated steps
                loss = (loss_ce + loss_dice) / accumulation_steps

            # scaled gradiants backward pass
            scaler.scale(loss).backward()

            # step the optimizer once accumulation target is hit
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                optimization_steps += 1 # track parameter updates

            # save training metrics
            train_loss += (loss.item() * accumulation_steps)
            preds = torch.argmax(outputs, dim=1)
            train_iou += calculate_iou(preds, targets)

            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / (optimization_steps * accumulation_steps)
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

        # update learning rate based on real validation loss behavior
        scheduler.step(avg_val_loss)

        print(
            f"\nMetrics:\n"
            f"Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f} || "
            f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}"
        )

        # save model tracking snapshots
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, "best_trail_model.pth")
            )
            print(f"New Best Validation IoU achieved! Checkpoint saved.")


if __name__ == "__main__":
    train_model()