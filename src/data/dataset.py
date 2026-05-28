# src/data/dataset.py

import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset

class MultimodalTrailDataset(Dataset):
    def __init__(self, config):
        # use global config.yaml for proper attr managing
        self.region_name = config['active_region']
        self.tile_size = config['tile_size']
        self.stride = config['stride']
        
        self.naip_path = config['naip_path']
        self.elev_path = config['elev_path']
        self.mask_path = config['mask_path']

        # open files briefly to read dimensions and compute global stats
        with rasterio.open(self.naip_path) as naip_src, rasterio.open(self.elev_path) as elev_src:
            self.height = naip_src.height
            self.width = naip_src.width
            
            # compute GLOBAL elevation bounds to preserve actual sloped topography across tiles
            print("Calculating global elevation statistics for accurate normalization...")

            # Read a lower resolution overview to speed up initialization on huge rasters
            elev_overview = elev_src.read(1, out_shape=(1024, 1024))
            self.global_elev_min = float(np.nanmin(elev_overview))
            self.global_elev_max = float(np.nanmax(elev_overview))
            print(f"Global Elevation Range: {self.global_elev_min}m to {self.global_elev_max}m")
        
        # pre-calc grid positions for tile slicing (sliding window)
        self.tiles = []
        for y in range(0, self.height - self.tile_size + 1, self.stride):
            for x in range(0, self.width - self.tile_size + 1, self.stride):
                self.tiles.append((x, y))
        
        # internal states for lazy loading
        self.naip_src = None       # 4 bands: R, G, B, NIR
        self.elev_src = None       # 1 band: Elevation (float32)
        self.mask_src = None       # 1 band: Ground truth (0 or 1)

    def _init_raster_workers(self):
        """
        lazily initialize file handlers inside the worker processes.
         - prevents file handler deadlock/corruption when num_workers > 0.
        """
        if self.naip_src is None:
            self.naip_src = rasterio.open(self.naip_path)
            self.elev_src = rasterio.open(self.elev_path)
            self.mask_src = rasterio.open(self.mask_path)
    
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):

        # for safe cross-process file streaming
        self._init_raster_workers()

        x_offset, y_offset = self.tiles[idx]
        
        # define window for reading
        window = rasterio.windows.Window(x_offset, y_offset, self.tile_size, self.tile_size)
        
        # read in NAIP data (bands, height, width) then convert to float32 and normalize
        naip_tile = self.naip_src.read(window=window).astype(np.float32) / 255.0
        
        # extract Red (band 1) and NIR (band 4) to calculate NDVI
        # NDVI formula: (NIR - Red) / (NIR + Red)
        red = naip_tile[0, :, :]
        nir = naip_tile[3, :, :]
        ndvi = (nir - red) / (nir + red + 1e-8) # 1e-8 prevents division by zero
        # rescale from [-1, 1] to [0, 1] to keep distributions uniform
        ndvi = (ndvi + 1.0) / 2.0
        ndvi = np.expand_dims(ndvi, axis=0)  # Shape: [1, H, W]
        
        # read elevation data and normalize 
        elev_tile = self.elev_src.read(1, window=window, boundless=True, fill_value=self.global_elev_min).astype(np.float32)

        # account for corrupted low nodata flags (i.e., -9999) from raw USGS metadata
        elev_tile[np.isnan(elev_tile)] = self.global_elev_min
        elev_tile[elev_tile < self.global_elev_min] = self.global_elev_min

        # if still issues with empty grid, reshape
        if elev_tile.shape[0] != self.tile_size or elev_tile.shape[1] != self.tile_size:
            # initialize using global_elev_min so padding normalizes to 0.0 after subtraction
            padded = np.full((self.tile_size, self.tile_size), self.global_elev_min, dtype=np.float32)
            h_limit = min(elev_tile.shape[0], self.tile_size)
            w_limit = min(elev_tile.shape[1], self.tile_size)
            padded[:h_limit, :w_limit] = elev_tile[:h_limit, :w_limit]
            elev_tile = padded

        # normalize using global extrema to retain uniform spatial gradients (slopes/cliffs)
        if self.global_elev_max > self.global_elev_min:
            elev_tile = (elev_tile - self.global_elev_min) / (self.global_elev_max - self.global_elev_min)
        else:
            elev_tile = np.zeros_like(elev_tile)

        # clamp to bounds to protect against edge interpolations outside original bounding boxes
        elev_tile = np.clip(elev_tile, 0.0, 1.0)
        elev_tile = np.expand_dims(elev_tile, axis=0)  # Shape: [1, H, W]
        
        # read in ground truth mask
        mask_tile = self.mask_src.read(1, window=window, boundless=True, fill_value=0).astype(np.int64)
        
        # concat inputs for symmetrical MAPA framework
        # visual/environmental vector: RGB + NIR + NDVI (5 channels total)
        visual_tensor = torch.from_numpy(np.concatenate([naip_tile, ndvi], axis=0))

        # elevation vector: normalized DEM (1 channel total)
        elevation_tensor = torch.from_numpy(elev_tile)
        
        target_tensor = torch.from_numpy(mask_tile).long() # Classification target
        
        return visual_tensor, elevation_tensor, target_tensor

    def __del__(self):
        # Safe clean closure only if files were opened in the parent process
        if getattr(self, 'naip_src', None) is not None:
            self.naip_src.close()
        if getattr(self, 'elev_src', None) is not None:
            self.elev_src.close()
        if getattr(self, 'mask_src', None) is not None:
            self.mask_src.close()