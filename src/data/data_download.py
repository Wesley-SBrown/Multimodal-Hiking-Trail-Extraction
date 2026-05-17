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

    # definite async export
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
    print("Download complete")


if __name__=='__main__':
    # OSM testing
    # place = "Mount Tamalpais State Park, California, USA"
    # gdf = download_osm_trails(place)

    # if gdf is not None:
    #     gdf.plot(linewidth=1.5, color='green', figsize=(8,8))
    #     plt.title(f"Trails in {place}")
    #     plt.show()

    auth() # run Earth Engine authentication 

    # NAIP testing
    # test bounding box from OSM matplot output
    mt_tam_bbox = [-122.625, 37.870, -122.580, 37.905]
    
    download_naip_images_to_drive(mt_tam_bbox)