# src/data/preprocessing.py

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import osmnx as ox
import numpy as np

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
        raster_crs = src.crs.to_string()
        transform = src.transform
        width = src.width
        height = src.height

    # pull from OSM using exact bounding box
    custom_filter = '["highway"~"path|footway|track"]'
    edges = None


    # TODO: WIP - not functioning properly need to fix somehow
    #           - might be a cache issue? unsure
    #           - maybe pivot to local data storage instead of re-querying 
    # backup: if API bounding box fails, use the robust place name query
    if place_name:
        print(f"Attempting robust fetch via place name: {place_name}...")
        try:
            graph = ox.graph_from_place(place_name, custom_filter=custom_filter)
            _, edges = ox.graph_to_gdfs(graph)
        except Exception as e:
            print(f"Place query failed: {e}. Falling back to bounding box...")

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
            print("💡 Tip: The public Overpass API is overloaded. Try running again in a few minutes,")
            print("or pass the explicit 'place_name' parameter to use a cached query.")
            return
    

    # perform projection of vector trails from degrees to distance meaursements
    print(f"Projecting trails to match GeoTIFF CRS: {raster_crs}...")
    edges_projected = edges.to_crs(raster_crs)

    # apply a spatial buffer to give the lines a realistic physical width
    print("Buffering vector trails into 2D trail polygons...")
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

    print(f"Saving final binary ground-truth mask to {output_mask_path}...")
    with rasterio.open(output_mask_path, 'w', **meta) as dst:
        dst.write(mask_array, 1)
        
    print("Mask alignment pipeline complete!")


if __name__ == "__main__":
    raw_tiff = "../../data/raw/mt_tamalpais_naip.tif"
    output_mask = "../../data/masks/mt_tamalpais_mask.tif"

    test_area = "Mount Tamalpais State Park, California, USA"
    generate_training_mask(raw_tiff, output_mask, place_name=test_area)