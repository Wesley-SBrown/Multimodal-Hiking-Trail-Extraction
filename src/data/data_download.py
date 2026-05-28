# src/data/data_download.py

import osmnx as ox
import matplotlib.pyplot as plt
import ee 
import os
import time
from src.utils.config_loader import load_region_config

# Open Street Map testing
def download_osm_trails(place_name: str):
    """
    Downloads paths for given location

    param: place_name
    """

    print(f"Searching Open Street Map for: {place_name}")

    path_filter = '["highway"~"path|footway|track"]'

    try:
        # strat A: parse as bounded polygon area (i.e., Mount Tam)
        graph = ox.graph_from_place(place_name, custom_filter=path_filter)
        _, edges = ox.graph_to_gdfs(graph)
        print(f"Found {len(edges)} trail segments via polygon boundary query.")
        return edges
    
    except Exception as e:
        print(f"Notice: Place query failed ({e}). Falling back to center-point bounding box...")
        try:
            # strat B: geocode to point and generate a 0.05 padded bounding box 
            # mirrors the coordinate matrix sent to Earth Engine
            lat, lon = ox.geocode(place_name)
            padding = 0.05
            
            # ox.graph_from_bbox expects (north, south, east, west)
            graph = ox.graph_from_bbox(
                bbox=(lat + padding, lat - padding, lon + padding, lon - padding),
                custom_filter=path_filter
            )
            _, edges = ox.graph_to_gdfs(graph)
            print(f"Found {len(edges)} trail segments via padded bounding box query.")
            return edges
            
        except Exception as fallback_error:
            print(f"Error: All methods failed to download trail network: {fallback_error}")
            return None

# Google Earth Engine Section

# Authenticate Earth Engine
def auth():
    try:
        ee.Initialize(project='hiking-trail-extraction')
    except Exception:
        ee.Authenticate() # only needed once
        ee.Initialize(project='hiking-trail-extraction')

# collect NAIP data
# files are too large for direct download, so have to use a gdrive setup (50MB limit)
def download_naip_images_to_drive(bbox, folder_name='ECS111_Trail_Data', region=None):
    """
    Downloads NAIP satellite images from Google Earth Engine to gdrive
    params: bbox: bounding box with lat and lon min/max vals
            folder_name: output gdrive folder location
    """

    print('Pulling NAIP data...')

    # define the region of interest from the bbox
    roi = ee.Geometry.Rectangle(bbox)

    # pull from NAIP, filetering by coords and date
    data = (ee.ImageCollection('USDA/NAIP/DOQQ')
            .filterBounds(roi)
            .filterDate('2020-01-01', '2024-01-01'))
    
    # use mosaic to combine the layers together into a seemless image
    image = data.mosaic().select(['R','G','B','N']).clip(roi)

    # check for required region param
    if not region:
        print('ERROR: Region MISSING')
        return
    
    # define async export
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f'{region} naip',
        folder=folder_name,
        fileNamePrefix=f'{region}_naip',
        scale=0.6,
        region=roi,
        fileFormat='GeoTIFF',
        maxPixels=1e9
    )
    
    # start export 
    task.start()
    print(f"Task started! Exporting 0.6m high-res imagery to your Google Drive folder: '{folder_name}'")

    # task progress monitoring
    while task.active():
        print(f"Current Status: {task.status()['state']}...")
        time.sleep(15)

    end = task.status()
    if end['state'] == 'COMPLETED':
        print("The GeoTIFF saved to gdrive.")
        print("Download and save to 'data/raw/' folder.")
    else:
        print(f"Error: export failed: {end.get('error_message')}")

# collect usgs elevation data from earth engine
def download_usgs_elevation(bbox, folder_name="ECS111_Trail_Data", region=None):
    """
    Saves USGS 3DEP Digital Elevation Model (DEM) data within the given bounding box
    Then saves to specified google drive folder 
    params: bbox: bounding box of selection area
            folder_name: output gdrive folder location
    """

    print(f'Starting download of USGS data to gdrive: {folder_name}...')

    # define region of interest
    roi = ee.Geometry.Rectangle(bbox)

    # pulls from the USGS 3DEP 1/3 arc-second dataset (~10m resolution) 
    # will resample to match our NAIP grid resolution
    elevation_collection = ee.ImageCollection('USGS/3DEP/10m_collection')
    elevation_dataset = (elevation_collection.filterBounds(roi)
                         .mosaic()
                         .select(['elevation']))

    # fit the global elevation model to specific bounding box
    elevation_fit = elevation_dataset.clip(roi)

    # check for required region param
    if not region:
        print('ERROR: Region MISSING')
        return
    
    # setup async export 
    task = ee.batch.Export.image.toDrive(
        image=elevation_fit,
        description= f'{region} elevation',
        folder=folder_name,
        fileNamePrefix=f'{region}_elevation',
        scale=0.6, # forced rescaling to match NAIP
        region=roi,
        fileFormat='GeoTIFF',
        maxPixels=1e9
    )

    # begin task
    task.start()
    print(f'Task started!\nExporting USGS Elevation data to gdrive folder: {folder_name}')
    print('Status Check:')

    while task.active():
        print(f"Current Status: {task.status()['state']}...")
        time.sleep(15)

    final_status = task.status()
    if final_status['state']=='COMPLETED':
        print(f"Sucessfully completed task!\nData is stored in: {folder_name}")
        print('Download and save to cwd to use!')

if __name__=='__main__':
    # load in config
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config = load_region_config(PROJECT_ROOT)

    # uncomment to pull path data from Open Street Map 
    place = config['place_name']
    region = config['active_region']

    gdf = download_osm_trails(place)

    if gdf is not None:
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        local_vector_backup = os.path.join(raw_dir, region) + ".geojson"
        print(f"Caching successful OSM trail geometries locally to: {local_vector_backup}")
        gdf.to_file(local_vector_backup, driver="GeoJSON")

        gdf.plot(linewidth=1.5, color='green', figsize=(8,8))
        plt.title(f"Trails in {place}")
        plt.show()
        

    print(f"Geocoding boundary for '{place}' to calculate bounding box...")
    dynamic_bbox = None

    try:
        # strat A: official boundary polygon
        area_gdf = ox.geocode_to_gdf(place)
        bounds = area_gdf.total_bounds
        dynamic_bbox = [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]
        print(f"Successfully calculated bounding box from polygon boundary: {dynamic_bbox}")
    except Exception as polygon_error:
        # if fails:
        print(f"Notice: Could not resolve a polygon boundary. Falling back to center-point padding...")
        
        try:
            # strat B: geocode to a point if strat A fails (i.e. for Yosemite Valley)
            lat, lon = ox.geocode(place)
            print(f"Found center coordinate for {place}: Lat {lat}, Lon {lon}")
            
            # pad roughly 0.05 degrees in each direction (~3.5 miles / 5.5 km wide box)
            # testing scale that easily stays under Earth Engine's limits
            padding = 0.05 
            dynamic_bbox = [
                float(lon - padding), # min_lon (West)
                float(lat - padding), # min_lat (South)
                float(lon + padding), # max_lon (East)
                float(lat + padding)  # max_lat (North)
            ]
            print(f"Generated a padded bounding box around center point: {dynamic_bbox}")
        except Exception as point_error:
            print(f"ERROR: Complete geocoding failure for '{place}': {point_error}")

    # run if either method worked
    if dynamic_bbox:
        auth() # run Earth Engine authentication 

        # download satellite imagery using the extracted coordinates
        download_naip_images_to_drive(dynamic_bbox, region=region)

        # download elevation data using the exact same extracted coordinates
        download_usgs_elevation(dynamic_bbox, region=region)