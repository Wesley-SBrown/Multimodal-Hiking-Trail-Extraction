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
    custom_filter = '["highway"~"path|footway|track"]'
    edges = None

    # define a local path to store paths
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    region = load_region_config(PROJECT_ROOT)['active_region']

    local_vector_backup = os.path.join(raw_dir, region) + ".geojson"

    if os.path.exists(local_vector_backup):
        print(f"Using backup at {local_vector_backup}. Loading offline...")
        edges = gpd.read_file(local_vector_backup)

    elif place_name:
        print(f"Local backup missing. Querying OSM via place name: {place_name}...")
        path_filter = '["highway"~"path|footway|track"]'
        try:
            graph = ox.graph_from_place(place_name, custom_filter=path_filter)
            _, edges = ox.graph_to_gdfs(graph)
            
            # save local copy
            print(f"Saving vector backup to: {local_vector_backup}...")
            edges.to_file(local_vector_backup, driver="GeoJSON")
        except Exception as e:
            print(f"ERROR: Place query failed: {e}")

    if edges is None:
        # OSMnx expects (north, south, east, west)
        # rasterio bounds are (left/west, bottom/south, right/east, top/north)
        north, south, east, west = bounds.top, bounds.bottom, bounds.right, bounds.left
        print(f"Querying OSM via bounding box: N:{north}, S:{south}, E:{east}, W:{west}...")
        
        try:
            graph = ox.graph_from_bbox(
                bbox=(north, south, east, west),
                custom_filter=custom_filter
            )
            _, edges = ox.graph_to_gdfs(graph)
        except Exception as e:
            print(f"ERROR: OSM Server rejected connection: {e}")
            print("Try passing the explicit 'place_name' parameter to use a cached query.")
            return
        
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


if __name__ == "__main__":
    # test with mt tam
    raw_tiff = "../../data/raw/mt_tamalpais_naip.tif"
    output_mask = "../../data/masks/mt_tamalpais_mask.tif"

    test_area = "Mount Tamalpais State Park, California, USA"
    generate_training_mask(raw_tiff, output_mask, place_name=test_area)