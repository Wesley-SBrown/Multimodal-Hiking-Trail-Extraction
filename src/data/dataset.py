# src/data/dataset.py

import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np

class MultimodalTrailDataset(Dataset):
    def __init__(self, naip_path, elev_path, mask_path, tile_size=512, stride=512):
        self.tile_size = tile_size
        self.stride = stride
        
        # open the raster files using rasterio
        self.naip_src = rasterio.open(naip_path)       # 4 bands: R, G, B, NIR
        self.elev_src = rasterio.open(elev_path)       # 1 band: Elevation (float32)
        self.mask_src = rasterio.open(mask_path)       # 1 band: Ground truth (0 or 1)
        
        self.height = self.naip_src.height
        self.width = self.naip_src.width
        
        # pre-calc grid positions for tile slicing (sliding window)
        self.tiles = []
        for y in range(0, self.height - tile_size + 1, stride):
            for x in range(0, self.width - tile_size + 1, stride):
                self.tiles.append((x, y))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        x_offset, y_offset = self.tiles[idx]
        
        # define window for reading
        window = rasterio.windows.Window(x_offset, y_offset, self.tile_size, self.tile_size)
        
        # read in NAIP data (bands, height, width) then convert to float32
        naip_tile = self.naip_src.read(window=window).astype(np.float32) / 255.0
        
        # extract Red (band 1) and NIR (band 4) to calculate NDVI
        # NDVI formula: (NIR - Red) / (NIR + Red)
        red = naip_tile[0, :, :]
        nir = naip_tile[3, :, :]
        ndvi = (nir - red) / (nir + red + 1e-8) # 1e-8 prevents division by zero
        ndvi = np.expand_dims(ndvi, axis=0)     # Shape: [1, H, W]
        
        # read elevation data and normalize 
        elev_tile = self.elev_src.read(1, window=window).astype(np.float32)

        # min-max scaling to force the values between 0.0 and 1.0
        elev_min, elev_max = elev_tile.min(), elev_tile.max()

        if elev_max > elev_min:
            elev_tile = (elev_tile - elev_min) / (elev_max - elev_min)
        else:
            elev_tile = np.zeros_like(elev_tile)
        elev_tile = np.expand_dims(elev_tile, axis=0) # Shape: [1, H, W]
        
        # read in ground truth mask
        mask_tile = self.mask_src.read(1, window=window).astype(np.float32)
        
        # concat inputs for symmetrical MAPA framework
        # visual/environmental vector: RGB + NIR + NDVI (5 channels total)
        visual_tensor = torch.from_numpy(np.concatenate([naip_tile, ndvi], axis=0))

        # elevation vector: normalized DEM (1 channel total)
        elevation_tensor = torch.from_numpy(elev_tile)
        
        target_tensor = torch.from_numpy(mask_tile).long() # Classification target
        
        return visual_tensor, elevation_tensor, target_tensor

    def __del__(self):
        # Clean closure of files when dataset is destroyed
        self.naip_src.close()
        self.elev_src.close()
        self.mask_src.close()