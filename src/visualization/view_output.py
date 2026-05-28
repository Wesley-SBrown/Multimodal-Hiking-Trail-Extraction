# src/visualization/visualize_output.py

import os
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt

# import dataset module
from src.data.dataset import MultimodalTrailDataset
from src.utils.config_loader import load_region_config

def plot_vectorized_results():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) if "__file__" in locals() else os.getcwd()
    config = load_region_config(PROJECT_ROOT)
    region = config['active_region']
    TILE = config['active_tile_id']

    naip_path = config["naip_path"]

    geojson_template = config.get("output_geojson", "data/output_extracted_trails_tile_{tile_id}.geojson")
    formatted_geojson = geojson_template.format(tile_id=TILE)
    geojson_path = os.path.join(PROJECT_ROOT, formatted_geojson)

    # confirm exists
    if not os.path.exists(geojson_path):
        print("ERROR: Extracted GeoJSON missing! Please run 'src.inference.reconstruct' first to generate the vectors.")
        return

    print("Reading extracted vector network...")
    gdf = gpd.read_file(geojson_path)

    # extract bounding parameters to focus the viewport
    dataset = MultimodalTrailDataset(config=config)
    x_offset, y_offset = dataset.tiles[TILE]
    
    print("Opening background NAIP satellite image context...")
    with rasterio.open(naip_path) as src:

        # Calculate bounding coordinates of just this tile
        tile_top_left = src.transform * (x_offset, y_offset)
        tile_bottom_right = src.transform * (x_offset + 512, y_offset + 512)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        show(src.read([1, 2, 3]), transform=src.transform, ax=ax, alpha=0.9)
        
        gdf.plot(ax=ax, color="#00FFFF", linewidth=3.0, label="Trained Extracted Network Topology")
        
        # crop the visualization box to focus exactly around the tile
        # note: affine matrix layouts can flip y coordinates depending on northern orientation
        ax.set_xlim([min(tile_top_left[0], tile_bottom_right[0]), max(tile_top_left[0], tile_bottom_right[0])])
        ax.set_ylim([min(tile_top_left[1], tile_bottom_right[1]), max(tile_top_left[1], tile_bottom_right[1])])
        
        ax.set_title("Inference Stage: Local GeoJSON Trail Vector Alignment (Tile Area)", fontsize=11, fontweight="bold")
        custom_line = plt.Line2D([0], [0], color="#00FFFF", lw=3.0, label='Extracted Continuous Vector Path')
        ax.legend(handles=[custom_line], loc='upper right')
        
        # save plot
        output_png = os.path.join(PROJECT_ROOT, f"data/{region}_tile_{TILE}_mapped.png")
        plt.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"Final vector overlay plot compiled and saved to:\n{output_png}")

        plt.show()
if __name__ == "__main__":
    plot_vectorized_results()