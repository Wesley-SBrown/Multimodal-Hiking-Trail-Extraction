# src/data/preprocessing.py

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import osmnx as ox
import numpy as np
import os
from src.utils.config_loader import load_region_config

# set a global cache storage location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ox.settings.cache_folder = os.path.join(PROJECT_ROOT, "cache")
ox.settings.use_cache = True

# define training mask function
def generate_training_mask(tiff_path, output_mask_path, place_name=None):
    """
    Reads in an NAIP GeoTiff, downloads the cooresponding OSM trails, applies a 2m spatial buffer,
    then rasterizes them into a matching binary mask
    """

    print(f"Reading in satellite dims from {tiff_path}")

    with rasterio.open(tiff_path) as src:
        meta = src.meta.copy()
        bounds = src.bounds
        raster_crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

    # pull from OSM using exact bounding box
    path_filter = '["highway"~"path|footway|track"]'
    edges = None

    # define a local path to store paths
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    region = load_region_config(PROJECT_ROOT)['active_region']

    local_vector_backup = os.path.join(raw_dir, region) + ".geojson"

    if os.path.exists(local_vector_backup):
        print(f"Using backup at {local_vector_backup}. Loading offline...")
        edges = gpd.read_file(local_vector_backup)

    elif place_name:
        # strat A: polygon graph by name
        print(f"Local backup missing. Querying OSM via place name: {place_name}...")
        try:
            graph = ox.graph_from_place(place_name, custom_filter=path_filter)
            _, edges = ox.graph_to_gdfs(graph)
            
        except Exception as e:
            # strat B: center point padding 
            try:
                lat, lon = ox.geocode(place_name)
                padding = 0.05
                graph = ox.graph_from_bbox(
                    bbox=(lat + padding, lat - padding, lon + padding, lon - padding),
                    custom_filter=path_filter
                )
                _, edges = ox.graph_to_gdfs(graph)
            except Exception as e2:
                print(f"ERROR: All OSM query attempts failed: {e2}")


    if edges is None:
        # OSMnx expects (north, south, east, west)
        # rasterio bounds are (left/west, bottom/south, right/east, top/north)
        north, south, east, west = bounds.top, bounds.bottom, bounds.right, bounds.left
        print(f"Querying OSM via bounding box: N:{north}, S:{south}, E:{east}, W:{west}...")
        
        try:
            graph = ox.graph_from_bbox(
                bbox=(north, south, east, west),
                custom_filter=path_filter
            )
            _, edges = ox.graph_to_gdfs(graph)
        except Exception as e:
            print(f"ERROR: OSM Server rejected connection: {e}")
            print("Try passing the explicit 'place_name' parameter to use a cached query.")
            raise RuntimeError("CRITICAL ERROR: Could not connect to OSM server and no offline backup exists. Mask pipeline aborted.")
        
    # stop the pipeline if no trails were recovered
    if edges is None or len(edges) == 0:
        raise ValueError("CRITICAL ERROR: No trail vectors could be retrieved from OSM. Cannot generate mask.")

    # save local vector backup
    if not os.path.exists(local_vector_backup) and edges is not None:
        print(f"Saving offline vector backup to: {local_vector_backup}...")
        edges.to_file(local_vector_backup, driver="GeoJSON")

    # check if CRS is geographic
    if raster_crs.is_geographic:
        utm_crs = edges.estimate_utm_crs()

        # convert to metric units
        edges_metric = edges.to_crs(utm_crs)

        # apply a spatial buffer to give the lines a realistic physical width
        buffered_metric = edges_metric.geometry.buffer(2.0)

        # project back to match GeoTiff crs
        buffered_trails = buffered_metric.to_crs(raster_crs)

    else:
        edges_projected = edges.to_crs(raster_crs)
        buffered_trails = edges_projected.geometry.buffer(2.0)

    # save the polygons to a binary 2D numpy array matching the image dimensions
    print("Rasterizing: Saving trail geometries into pixel mask...")
    shapes = [(geom, 1) for geom in buffered_trails if not geom.is_empty]

    # rasterize: returns 1 if intersection
    mask_array = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # save mask as a single band GeoTiff
    meta.update(dtype=rasterio.uint8, count=1, nodata=None)
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

    print(f"Saving final binary ground-truth mask to {output_mask_path}...")
    with rasterio.open(output_mask_path, 'w', **meta) as dst:
        dst.write(mask_array, 1)
        
    print("Mask alignment pipeline complete!")

def log_valid_trail_tiles(dataset, output_log_path):
    """
    Iterates through the dataset grid, detects tiles that contain positive 
    ground-truth trail labels, and saves their index IDs to a text file.
    """
    print("Scanning dataset grid for tiles containing active trail segments...")
    valid_ids = []
    
    for idx in range(len(dataset)):
        # dataset[idx] returns (visual, elevation, mask)
        _, _, mask = dataset[idx]
        
        # if the ground truth mask has any trail pixels (value > 0)
        if (mask > 0).any():
            valid_ids.append(idx)
            
    # ensure directory exists and write the IDs out
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    with open(output_log_path, "w") as f:
        for tile_id in valid_ids:
            f.write(f"{tile_id}\n")
            
    print(f"Success! Found {len(valid_ids)} valid trail tiles out of {len(dataset)} total segments.")
    print(f"Valid tile IDs cached to: {output_log_path}")

if __name__ == "__main__":
    print("Loading project configuration...")
    config = load_region_config(PROJECT_ROOT)
    region = config["active_region"]
    
    # extract configuration values
    raw_tiff = os.path.join(PROJECT_ROOT, "data/raw", region + "_naip.tif")
    output_mask = os.path.join(PROJECT_ROOT, "data/masks", config["active_region"] + "_mask.tif")
    test_area = config["place_name"]
    
    # uncomment to generate/regenerate the base mask first
    # generate_training_mask(raw_tiff, output_mask, place_name=test_area)

    from src.data.dataset import MultimodalTrailDataset

    print(f"Initializing Multimodal Trail Dataset for region: {region}...")
    dataset = MultimodalTrailDataset(config=config)

    # extract the log path from your updated config.yaml
    inference_cfg = config.get("inference", {})
    output_log_path = inference_cfg.get("valid_tiles_log", f"outputs/{region}_valid_tiles.txt")
    full_log_path = os.path.join(PROJECT_ROOT, output_log_path)

    # run the function to scan and log valid tiles
    log_valid_trail_tiles(dataset, full_log_path)