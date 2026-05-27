# test_pipeline.py

import os
import torch
from torch.utils.data import DataLoader
from src.data.preprocessing import generate_training_mask
from src.data.dataset import MultimodalTrailDataset
from src.utils.config_loader import load_region_config

# create sanity check function to make sure current pipeline is functioning
def run_sanity_check():
    print("Setting up directory paths")
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "masks")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # pull active region paths from config/regions.yaml
    config = load_region_config(PROJECT_ROOT)

    naip_path = config["naip_path"]
    elev_path = config["elev_path"]
    mask_path = config["mask_path"]

    test_area = config["place_name"]
    tile_size = config["tile_size"]
    stride = config["stride"]

    print(f"Active region: {config['active_region']}")
    print(f"Place name: {test_area}")
    print(f"Tile size: {tile_size}, Stride: {stride}")

    print("Checking data requirements ===")

    # check if local data exists already
    # required because earth engine data is saved to drive first
    if not os.path.exists(naip_path) or not os.path.exists(elev_path):
        print("ERROR: Missing source GeoTIFF files inside data/raw/!")
        print("Please ensure you run data_download.py or export the files from Google Earth Engine")
        print(f"Expected NAIP file:\n{naip_path}")
        print(f"Expected elevation file:\n{elev_path}")
        return

    print("Source images found! Preprocessing...")

    print("\nTesting label extraction & rasterization")

    try:
        generate_training_mask(naip_path, mask_path, place_name=test_area)
        print("Ground truth mask successfully created!")
    except Exception as e:
        print(f"ERROR: Mask Generation Failed: {e}")
        return

    print("\nInstantiating multimodal dataset")
    try:
        dataset = MultimodalTrailDataset(config=config)
        print(f"Dataset built successfully! Total parsed tiles: {len(dataset)}")
    except Exception as e:
        print(f"ERROR: Dataset initialization failed: {e}")
        return

    print("\nTesting Parallel Processing Data Loader")

    # num_workers=2 to explicitly test the lazy-loading fix for cross-process issues
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    try:
        # pull exactly one batch out of the generator stream
        visual_batch, elev_batch, target_batch = next(iter(dataloader))
        
        print("\nCOMPONENT CHECKS")
        print(f"Visual Tensor (RGB+NIR+NDVI) Shape : {visual_batch.shape}  -> Expected: [B, 5, {tile_size}, {tile_size}]")
        print(f"Elevation Tensor (DEM Data) Shape: {elev_batch.shape}  -> Expected: [B, 1, {tile_size}, {tile_size}]")
        print(f"Target Mask (Ground Truth) Shape : {target_batch.shape}  -> Expected: [B, {tile_size}, {tile_size}]")
        
        print("\nValue Range & Sanity Check")
        print(f"Visual range   : Min={visual_batch.min().item():.4f}, Max={visual_batch.max().item():.4f} \
               (Expected: ~0.0 to 1.0)")
        print(f"Elevation range: Min={elev_batch.min().item():.4f}, Max={elev_batch.max().item():.4f} \
              (Expected: 0.0 to 1.0 Global Normalization)")
        print(f"Unique Labels  : {torch.unique(target_batch).tolist()} (Expected: [0, 1] representing Non-Trail / Trail)")
        
        if visual_batch.shape[1] == 5 and elev_batch.shape[1] == 1:
            print("\nEverything is perfectly aligned! Your data pipeline is ready for model injection.")
        else:
            print("\nERROR: Vector dimensions mismatched. Check your tensor concatenation layers.")

    except Exception as e:
        print(f"ERROR: DataLoader extraction failed. Worker collision check: {e}")

if __name__ == "__main__":
    run_sanity_check()