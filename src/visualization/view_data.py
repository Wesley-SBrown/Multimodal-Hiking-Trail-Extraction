# src/visualization/view_data.py

import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path so src imports work when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import MultimodalTrailDataset
from src.utils.config_loader import load_region_config

def plot_multimodal_tile(idx=None):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) if "__file__" in locals() else os.getcwd()
    
    # attempt to pull local data from the active region listed in config/regions.yaml
    region_config = load_region_config(PROJECT_ROOT)

    mask_path = region_config["mask_path"]

    tile_size = region_config["tile_size"]
    stride = region_config["stride"]

    print(f"Active region: {region_config['active_region']}")
    print(f"Place name: {region_config['place_name']}")
    print(f"Tile size: {tile_size}, Stride: {stride}")

    # check if mask is missing
    if not os.path.exists(mask_path):
        print("ERROR: Masks missing. Please execute test_pipeline.py first to compile the data layers!")
        return

    # instantiate dataset locally
    dataset = MultimodalTrailDataset(config=region_config)
    
    # search for a tile that contains at least some trail pixels so the visualization isn't empty
    if idx is None:
        print("Searching for a tile intersecting a physical hiking trail...")
        found_good_tile = False

        # Seed pseudo-random search for consistency
        np.random.seed(67)
        search_indices = np.random.permutation(len(dataset))
        
        for sample_idx in search_indices:
            _, _, target = dataset[sample_idx]
            if torch.sum(target) > 50:  # require at least 50 trail pixels
                idx = int(sample_idx)
                found_good_tile = True
                break
        
        if not found_good_tile:
            idx = 0
            print("WARNING: No trails found in searched tiles. Defaulting to index 0.")
            
    print(f"Extracting Data Stream Elements from Tile Index: {idx}")
    visual_tensor, elevation_tensor, target_tensor = dataset[idx]

    # decode the multi modal channels for use in matplotlib
    # first extract RGB channels (Bands 0, 1, 2) and transpose from [C, H, W] to [H, W, C]
    rgb_image = visual_tensor[0:3, :, :].numpy().transpose(1, 2, 0)

    # clip pixel overflow spikes to protect plot contrast
    rgb_image = np.clip(rgb_image, 0.0, 1.0)

    # next extract NDVI channel (Band 4) -> Shape: [H, W]
    ndvi_map = visual_tensor[4, :, :].numpy()

    # next extract Elevation (DEM) -> Shape: [H, W]
    elevation_map = elevation_tensor[0, :, :].numpy()

    # finally extract Ground Truth Mask -> Shape: [H, W]
    ground_truth = target_tensor.numpy()

    # create grid for the plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    
    # Panel A: raw colored satellite image
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("1. True Color RGB (NAIP Satellite)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel B: normalized difference vegetation index
    im_ndvi = axes[0, 1].imshow(ndvi_map, cmap="YlGn")
    axes[0, 1].set_title("2. NDVI (Vegetation Density Mask)", fontsize=11, fontweight="bold")
    fig.colorbar(im_ndvi, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Density Index")
    axes[0, 1].axis("off")

    # Panel C: topology map 
    im_elev = axes[1, 0].imshow(elevation_map, cmap="terrain")
    axes[1, 0].set_title("3. Topographic DEM (USGS Elevation)", fontsize=11, fontweight="bold")
    fig.colorbar(im_elev, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Normalized Height")
    axes[1, 0].axis("off")

    # Panel D: Ground Truth Path Overlaid on top of Satellite
    axes[1, 1].imshow(rgb_image)

    # create a red overlay for the vector tracks
    masked_truth = np.ma.masked_where(ground_truth == 0, ground_truth)
    axes[1, 1].imshow(masked_truth, cmap="Set1", alpha=0.75, interpolation="none")
    axes[1, 1].set_title("4. Ground Truth Mask (OSM Vector Overlay)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    plt.suptitle(
        f"Multimodal Data Alignment Grid | Region: {region_config['active_region']} | Tile ID: {idx}",
        fontsize=14,
        fontweight="bold",
        y=0.96
    )
    plt.tight_layout()
    
    # save output pic locally in data folder
    output_png = os.path.join(
        PROJECT_ROOT,
        "data",
        f"{region_config['active_region']}_tile_{idx}.png"
    )
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Visualization plot successfully compiled and saved to:\n{output_png}")
    plt.show()

if __name__ == "__main__":
    plot_multimodal_tile(200)