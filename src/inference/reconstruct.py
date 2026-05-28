# src/inference/reconstruct.py 

import os
os.environ['GDAL_DATA'] = r'C:\Users\flyin\Miniconda3\envs\hike\Lib\site-packages\osgeo\data\gdal'
import torch
import numpy as np
import rasterio
from skimage.morphology import skeletonize, remove_small_objects, closing, disk
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString

# import previous modules
from src.models.trail_net import MultiModalNet
from src.data.dataset import MultimodalTrailDataset
from src.utils.config_loader import load_region_config


def mask_to_graph(pred_mask, transform, crs, x_offset, y_offset, disk_radius=2.2, min_pixel_length=15):    
    """
    Collapses a binary pixel segmentation mask into a skeleton & extracts coord nodes
    Exports into vectored GeoDataFrame
    """

    # use morphological closing with small radius - helps with gaps
    closed_mask = closing(pred_mask > 0, footprint=disk(disk_radius)) # larger disk(x) value reaches farther

    # clean isolated stray noise blobs before building the topology graph
    clean_mask = remove_small_objects(closed_mask > 0, min_size=min_pixel_length)

    # thin out the mask into spine - single pixl wide
    skeleton = skeletonize(clean_mask > 0).astype(np.uint8)

    # extract coords from the skeleton
    y_coords, x_coords = np.where(skeleton > 0)

    if len(x_coords) == 0:
        print("Warning: No trails found in this section")
        return None
    
    # set up spatial graph using the 8 way pixel connectivity 
    pixel_graph = nx.Graph()
    points = list(zip(x_coords, y_coords))
    points_unique = set(points)

    # build out graph
    for (x,y) in points:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in points_unique:
                    pixel_graph.add_edge((x,y), neighbor)


    # extract the connected line segments then project onto geographic plane
    lines = []
    for component in nx.connected_components(pixel_graph):
        subgraph = pixel_graph.subgraph(component)

        # Filter out tiny residual segments to ensure clean paths
        if len(subgraph.nodes()) < min_pixel_length:
            continue

        # determine the endpoints of the trail
        # endpoints have degree of 1
        endpoints = [node for node in subgraph.nodes() if subgraph.degree(node) == 1]

        # define fallback in case of closed loop
        start_node = endpoints[0] if endpoints else list(subgraph.nodes())[0]

        # traverse the paths to building the lines
        dfs_edges = list(nx.dfs_edges(subgraph, source=start_node))

        # move on if there are no lines
        if not dfs_edges:
            continue

        # extract pixel locations in sequence 
        pixel_seq = [dfs_edges[0][0]] + [e[1] for e in dfs_edges]

        # use affine transformation matrix to map to geospatial coords
        geo_coords = [transform * (px + x_offset, py + y_offset) for px, py in pixel_seq]

        if len(geo_coords) >= 2:
            lines.append(LineString(geo_coords))

    # check if lines were captured 
    if not lines:
        return None
    
    # combine into geodf
    gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
    return gdf

def run_inference():
    # define paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    config = load_region_config(PROJECT_ROOT)
    TILE = config.get("active_tile_id")
    
    naip_path = config["naip_path"]
    elev_path = config["elev_path"]
    mask_path = config["mask_path"]

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints/best_trail_model.pth")

    geojson_template = config.get("output_geojson", "data/output_extracted_trails_tile_{tile_id}.geojson")
    formatted_geojson = geojson_template.format(tile_id=TILE)
    output_geojson = os.path.join(PROJECT_ROOT, formatted_geojson)

    # check for gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and pull sample validation tileDataset(
    dataset = MultimodalTrailDataset(config=config)
    visual, elev, _ = dataset[TILE] # test tile

    # extract the exact tile offset from the dataset's tracking grid
    x_offset, y_offset = dataset.tiles[TILE]

    # load trained network weights
    model = MultiModalNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))    
    model.eval()

    print("Running multi-modal forward inference pass...")
    with torch.no_grad():
        # add batch dimension [1, C, H, W]
        vis_in = visual.unsqueeze(0).to(device)
        elev_in = elev.unsqueeze(0).to(device)
        
        outputs = model(vis_in, elev_in)

        # extract continuous trail probabilities
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        trail_probs = probs[1, :, :]  # Class 1 probability map [512, 512]
        
        # set soft activation threshold to keep the network connected
        CONFIDENCE_THRESHOLD = config['confidence_threshold']
        binary_preds = (trail_probs >= CONFIDENCE_THRESHOLD).astype(np.uint8)

    # extract spatial parameters to properly anchor lines onto global maps
    with rasterio.open(naip_path) as src:
        transform = src.transform
        crs = src.crs

    print("Collapsing pixel predictions into topological graph paths...")

    disk_radius = config.get("morphology_disk_radius")
    min_length = config.get("min_pixel_length")

    extracted_gdf = mask_to_graph(binary_preds, transform, crs, x_offset, y_offset, 
                                  disk_radius=disk_radius, min_pixel_length=min_length)

    if extracted_gdf is not None:
        os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
        extracted_gdf.to_file(output_geojson, driver="GeoJSON")
        print(f"SUCCESS! Vectorized trail network graphs generated and stored at:\n{output_geojson}")
    else:
        print("ERROR: Model inference completed, but topology construction failed to yield continuous paths.")

if __name__ == "__main__":
    run_inference()



