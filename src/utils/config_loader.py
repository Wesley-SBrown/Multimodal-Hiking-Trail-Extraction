# src/utils/config_loader.py
import os
import yaml


def load_region_config(project_root, config_path="config/config.yaml"):
    """
    Loads the active region settings from config/regions.yaml.

    This lets us switch regions without hardcoding paths in every script.
    """

    full_config_path = os.path.join(project_root, config_path)

    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"Missing region config file: {full_config_path}")

    with open(full_config_path, "r") as file:
        config = yaml.safe_load(file)

    active_region = config["active_region"]
    region_config = config["regions"][active_region]

    raw_dir = os.path.join(project_root, "data", "raw")
    mask_dir = os.path.join(project_root, "data", "masks")

    naip_path = os.path.join(raw_dir, region_config["naip_file"])
    elev_path = os.path.join(raw_dir, region_config["elev_file"])
    mask_path = os.path.join(mask_dir, region_config["mask_file"])

    hyperparameter_config = config["hyperparameters"]
    tile_size = hyperparameter_config['tile_size']
    stride = hyperparameter_config['stride']

    return {
        "active_region": active_region,
        "place_name": region_config["place_name"],
        "naip_path": naip_path,
        "elev_path": elev_path,
        "mask_path": mask_path,
        "tile_size": tile_size,
        "stride": stride,
    }