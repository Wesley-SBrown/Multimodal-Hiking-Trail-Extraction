import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from skimage.morphology import skeletonize
from skimage.measure import label

# add project root to Python path to ensure module resolution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import MultimodalTrailDataset
from src.models.trail_net import MultiModalNet
from src.utils.config_loader import load_region_config

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_trail_model.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
BATCH_SIZE = 1

os.makedirs(VIS_DIR, exist_ok=True)


# Metrics
def computeIoU(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-8)


def computeF1(pred, target):
    # flatten since computing pixel-wise classification targets
    return f1_score(target.flatten(), pred.flatten(), zero_division=0)


def computePixelAccuracy(pred, target):
    return (pred == target).sum() / pred.size


# thin trail eval
def createThinMask(mask):
    # skeletonize approximates the thin centerline of a trail
    return skeletonize(mask > 0).astype(np.uint8)


def computeThinTrailIoU(pred, target):
    # compare skeletonized prediction against a skeletonized ground truth target
    return computeIoU(createThinMask(pred), createThinMask(target))


# connectivity
def computeConnectivityScore(pred, target):
    # penalizes fragmentation: score is 1.0 when component counts match,
    # decreases as the difference grows
    pred_components = label(pred).max()
    target_components = label(target).max()
    return 1 / (1 + abs(pred_components - target_components))


# Inference with latency + VRAM tracking 
def measureInference(model, visual, elevation):
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    with torch.no_grad():
        # match mixed-precision optimization used during loop training steps
        with torch.amp.autocast_mode.autocast("cuda" if DEVICE.type == "cuda" else "cpu"):
            output = model(visual, elevation)
    latency = time.time() - start

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if DEVICE.type == "cuda" else 0

    return output, latency, peak_vram


# visualization
def saveVisualization(visual_tensor, mask_tensor, pred_tensor, index):
    # extract RGB channels (First 3 channels of your 5-channel visual tensor)
    image = visual_tensor.squeeze().cpu().numpy()[:3, :, :]
    mask = mask_tensor.squeeze().cpu().numpy()
    pred = pred_tensor.squeeze().cpu().numpy()

    # transpose back to HWC so matplotlib handles it properly
    image = np.transpose(image, (1, 2, 0))
    # clamp visual ranges safely
    image = np.clip(image, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Satellite Image (RGB)")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Model Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"sample_eval_{index}.png"))
    plt.close()


# main eval
def evaluate():
    print(f"Loading environment configurations from root space...")
    config = load_region_config(PROJECT_ROOT)

    region = config['active_region']
    
    print(f"Initializing MultiModalNet on device target: {DEVICE}")
    model = MultiModalNet(num_classes=2).to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Missing weight tracking state: {CHECKPOINT_PATH}. Run training first.")
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # re-instantiate base tracking dataset 
    full_dataset = MultimodalTrailDataset(config=config)

    # apply the same seed splitting architecture used in train.py to pull validation targets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # secure reproduction mapping via static seed generator
    torch.manual_seed(42)
    _, val_set = torch.utils.data.random_split(
        dataset=full_dataset,
        lengths=[train_size, val_size]
    )

    # evaluate validation subset using a clean batch architecture
    loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Evaluating model parameters across {len(loader)} holdout validation frames...")

    iouScores, f1Scores, pixelAccuracies = [], [], []
    thinTrailScores, connectivityScores = [], []
    latencies, vrams = [], []

    for index, (visual, elevation, mask) in enumerate(loader):
        visual = visual.to(DEVICE)
        elevation = elevation.to(DEVICE)
        mask = mask.to(DEVICE)

        output, latency, peakVRAM = measureInference(model, visual, elevation)

        # parse 2-channel logit maps into flat predictions using argmax
        pred = torch.argmax(output, dim=1)

        predNP = pred.squeeze().cpu().numpy().astype(np.uint8)
        maskNP = mask.squeeze().cpu().numpy().astype(np.uint8)

        # run morphological mapping arrays
        iou = computeIoU(predNP, maskNP)
        f1 = computeF1(predNP, maskNP)
        pixelAcc = computePixelAccuracy(predNP, maskNP)
        thinIoU = computeThinTrailIoU(predNP, maskNP)
        connectivity = computeConnectivityScore(predNP, maskNP)

        # Only track trail metrics if the ground truth actually contains a trail
        if maskNP.sum() > 0:
            iouScores.append(iou)
            f1Scores.append(f1)
            thinTrailScores.append(thinIoU)
            connectivityScores.append(connectivity)
        else:
            # If the tile is empty and the model correctly predicted empty:
            if predNP.sum() == 0:
                pixelAccuracies.append(pixelAcc) # Reward it for correct background mapping

        latencies.append(latency)
        vrams.append(peakVRAM)

        # save evaluation visualization file every 10 steps to prevent disk bloating
        if index % 10 == 0:
            saveVisualization(visual, mask, pred, index)
            
        print(f"[{index+1}/{len(loader)}] Frame IoU={iou:.4f} | Frame F1={f1:.4f} | Centerline IoU={thinIoU:.4f}")

    results = {
        "metadata": {
            "region":           region,
        },
        "metrics": {
            "mIoU":                  float(np.mean(iouScores)),
            "F1":                    float(np.mean(f1Scores)),
            "PixelAccuracy":         float(np.mean(pixelAccuracies)),
            "ThinTrailIoU":          float(np.mean(thinTrailScores)),
            "ConnectivityScore":     float(np.mean(connectivityScores)),
            "AverageLatencySeconds": float(np.mean(latencies)),
            "AverageVRAM_GB":        float(np.mean(vrams)),
        }
    }

    print("\nFINAL EVALUATION RESULTS")

    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSuccessfully stored evaluation analytics log at: {results_path}")


if __name__ == "__main__":
    evaluate()