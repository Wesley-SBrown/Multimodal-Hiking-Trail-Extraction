import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from skimage.morphology import skeletonize
from skimage.measure import label

# --- Config ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/model.pt"
OUTPUT_DIR = "outputs"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
BATCH_SIZE = 1
THRESHOLD = 0.5

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# CHANGE THESE IMPORTS TO MATCH YOUR PROJECT
from src.models.trail_net import TrailModel
from datasets.dataset import TrailDataset


# --- Metrics ---

def computeIoU(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-8)


def computeF1(pred, target):
    return f1_score(target.flatten(), pred.flatten())


def computePixelAccuracy(pred, target):
    return (pred == target).sum() / pred.size


# --- Thin Trail Evaluation ---

def createThinMask(mask):
    # Skeletonize approximates the thin centerline of a trail
    return skeletonize(mask > 0).astype(np.uint8)


def computeThinTrailIoU(pred, target):
    return computeIoU(pred, createThinMask(target))


# --- Connectivity ---

def computeConnectivityScore(pred, target):
    # Penalizes fragmentation: score is 1.0 when component counts match,
    # and decreases as the difference grows
    pred_components = label(pred).max()
    target_components = label(target).max()
    return 1 / (1 + abs(pred_components - target_components))


# --- Inference with latency + VRAM tracking ---

def measureInference(model, image):
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    with torch.no_grad():
        output = model(image)
    latency = time.time() - start

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0

    return output, latency, peak_vram


# --- Visualization ---

def saveVisualization(image, mask, pred, index):
    image = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image);        axes[0].set_title("Satellite Image")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Ground Truth")
    axes[2].imshow(pred, cmap="gray"); axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"sample_{index}.png"))
    plt.close()


# --- Main Evaluation ---

def evaluate():
    model = TrailModel()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    loader = DataLoader(TrailDataset(split="test"), batch_size=BATCH_SIZE, shuffle=False)

    iouScores, f1Scores, pixelAccuracies = [], [], []
    thinTrailScores, connectivityScores = [], []
    latencies, vrams = [], []

    for index, (image, mask) in enumerate(loader):
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        output, latency, peakVRAM = measureInference(model, image)

        pred = (torch.sigmoid(output) > THRESHOLD).float()

        predNP = pred.squeeze().cpu().numpy().astype(np.uint8)
        maskNP = mask.squeeze().cpu().numpy().astype(np.uint8)

        iou = computeIoU(predNP, maskNP)
        f1 = computeF1(predNP, maskNP)
        pixelAcc = computePixelAccuracy(predNP, maskNP)
        thinIoU = computeThinTrailIoU(predNP, maskNP)
        connectivity = computeConnectivityScore(predNP, maskNP)

        iouScores.append(iou)
        f1Scores.append(f1)
        pixelAccuracies.append(pixelAcc)
        thinTrailScores.append(thinIoU)
        connectivityScores.append(connectivity)
        latencies.append(latency)
        vrams.append(peakVRAM)

        saveVisualization(image, mask, pred, index)
        print(f"[{index+1}/{len(loader)}] IoU={iou:.4f} F1={f1:.4f}")

    results = {
        "mIoU":                  float(np.mean(iouScores)),
        "F1":                    float(np.mean(f1Scores)),
        "PixelAccuracy":         float(np.mean(pixelAccuracies)),
        "ThinTrailIoU":          float(np.mean(thinTrailScores)),
        "ConnectivityScore":     float(np.mean(connectivityScores)),
        "AverageLatencySeconds": float(np.mean(latencies)),
        "AverageVRAM_GB":        float(np.mean(vrams)),
    }

    print("\n==========================")
    print("FINAL EVALUATION RESULTS")
    print("==========================")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    evaluate()