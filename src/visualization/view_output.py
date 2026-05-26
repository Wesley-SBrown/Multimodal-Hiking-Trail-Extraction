# src/visualization/visualize_output.py

import os
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt

# import dataset module
from src.data.dataset import MultimodalTrailDataset

def plot_vectorized_results():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) if "__file__" in locals() else os.getcwd()

    naip_path = os.path.join(PROJECT_ROOT, "data/raw/mt_tamalpais_naip.tif")
    elev_path = os.path.join(PROJECT_ROOT, "data/raw/mt_tamalpais_elevation.tif")
    mask_path = os.path.join(PROJECT_ROOT, "data/masks/mt_tamalpais_mask.tif")
    geojson_path = os.path.join(PROJECT_ROOT, "data/output_extracted_trails.geojson")

    # confirm exists
    if not os.path.exists(geojson_path):
        print("ERROR: Extracted GeoJSON missing! Please run 'src.inference.reconstruct' first to generate the vectors.")
        return

    print("Reading extracted vector network...")
    gdf = gpd.read_file(geojson_path)

    # extract bounding parameters of Tile 424 to focus the viewport
    dataset = MultimodalTrailDataset(naip_path, elev_path, mask_path, tile_size=512, stride=256)
    x_offset, y_offset = dataset.tiles[424]
    
    print("Opening background NAIP satellite image context...")
    with rasterio.open(naip_path) as src:

        # Calculate bounding coordinates of just this tile
        tile_top_left = src.transform * (x_offset, y_offset)
        tile_bottom_right = src.transform * (x_offset + 512, y_offset + 512)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        show(src.read([1, 2, 3]), transform=src.transform, ax=ax, alpha=0.9)
        
        gdf.plot(ax=ax, color="#00FFFF", linewidth=3.0, label="Trained Extracted Network Topology")
        
        # crop the visualization box to focus exactly around Tile 424 - example tile
        # note: affine matrix layouts can flip y coordinates depending on northern orientation
        ax.set_xlim([min(tile_top_left[0], tile_bottom_right[0]), max(tile_top_left[0], tile_bottom_right[0])])
        ax.set_ylim([min(tile_top_left[1], tile_bottom_right[1]), max(tile_top_left[1], tile_bottom_right[1])])
        
        ax.set_title("Inference Stage: Local GeoJSON Trail Vector Alignment (Tile 424 Area)", fontsize=11, fontweight="bold")
        custom_line = plt.Line2D([0], [0], color="#00FFFF", lw=3.0, label='Extracted Continuous Vector Path')
        ax.legend(handles=[custom_line], loc='upper right')
        
        # save plot
        output_png = os.path.join(PROJECT_ROOT, "data/final_vector_overlay.png")
        plt.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"Final vector overlay plot compiled and saved to:\n{output_png}")

        plt.show()
if __name__ == "__main__":
    plot_vectorized_results()