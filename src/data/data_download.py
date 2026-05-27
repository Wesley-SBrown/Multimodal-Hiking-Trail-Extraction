# src/data/data_download.py

import osmnx as ox
import matplotlib.pyplot as plt
import ee 
import geemap
import os
import time

# Open Street Map testing
def download_osm_trails(place_name: str):
    """
    Downloads paths for given location

    param: place_name
    """

    print(f"Searching Open Street Map for: {place_name}")

    path_filter = '["highway"~"path|footway|track"]'

    try:
        # pull data as a network graph
        graph = ox.graph_from_place(place_name, custom_filter=path_filter)

        # convert to a geo dataframe
        nodes, edges = ox.graph_to_gdfs(graph)

        # if successful:
        print(f"Found {len(edges)} trail segments")
        return edges
    
    except Exception as e:
        print(f"Error: Failed to download data: {e}")

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
def download_naip_images_to_drive(bbox, folder_name='ECS111_Trail_Data'):
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

    # define async export
    task = ee.batch.Export.image.toDrive(
        image=image,
        description='mt_tamalpais_naip_highres',
        folder=folder_name,
        fileNamePrefix='mt_tamalpais_naip',
        scale=0.6,
        region=roi,
        fileFormat='GeoTIFF'
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
def download_usgs_elevation(bbox, folder_name="ECS111_Trail_Data"):
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
                         .select(['elevation'])
                         .mosaic()
                         .resample('bicubic'))

    # fit the global elevation model to specific bounding box
    elevation_fit = elevation_dataset.clip(roi)

    # setup async export 
    task = ee.batch.Export.image.toDrive(
        image=elevation_fit,
        description='mt_tamalpais_elevation',
        folder=folder_name,
        fileNamePrefix='mt_tamalpais_elevation',
        scale=0.6, # forced rescaling to match NAIP
        region=roi,
        fileFormat='GeoTIFF'
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
    # uncomment to pull path data from Open Street Map 
    place = "Mount Tamalpais State Park, California, USA"
    gdf = download_osm_trails(place)

    if gdf is not None:
        gdf.plot(linewidth=1.5, color='green', figsize=(8,8))
        plt.title(f"Trails in {place}")
        plt.show()

    auth() # run Earth Engine authentication 

    # NAIP testing
    # test bounding box from OSM matplot output
    mt_tam_bbox = [-122.625, 37.870, -122.580, 37.905]
    
    # uncomment to pull satellite data from NAIP
    download_naip_images_to_drive(mt_tam_bbox)

    # uncomment to pull elevation data from USGS
    download_usgs_elevation(mt_tam_bbox)