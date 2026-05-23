# src/data/dataset.py

import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np

class MultimodalTrailDataset(Dataset):
    def __init__(self, naip_path, elev_path, mask_path, tile_size=512, stride=512):
        self.tile_size = tile_size
        self.stride = stride
        
        self.naip_path = naip_path
        self.elev_path = elev_path
        self.mask_path = mask_path

        # open files briefly to read dimensions and compute global stats
        with rasterio.open(self.naip_path) as naip_src, rasterio.open(self.elev_path) as elev_src:
            self.height = naip_src.height
            self.width = naip_src.width
            
            # compute GLOBAL elevation bounds to preserve actual sloped topography across tiles
            print("Calculating global elevation statistics for accurate normalization...")

            # Read a lower resolution overview to speed up initialization on huge rasters
            elev_overview = elev_src.read(1, out_shape=(1024, 1024))
            self.global_elev_min = float(elev_overview.min())
            self.global_elev_max = float(elev_overview.max())
            print(f"Global Elevation Range: {self.global_elev_min}m to {self.global_elev_max}m")
        
        # pre-calc grid positions for tile slicing (sliding window)
        self.tiles = []
        for y in range(0, self.height - tile_size + 1, stride):
            for x in range(0, self.width - tile_size + 1, stride):
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
        elev_tile = self.elev_src.read(1, window=window).astype(np.float32)

        # normalize using global extrema to retain uniform spatial gradients (slopes/cliffs)
        if self.global_elev_max > self.global_elev_min:
            elev_tile = (elev_tile - self.global_elev_min) / (self.global_elev_max - self.global_elev_min)
        else:
            elev_tile = np.zeros_like(elev_tile)

        # clamp to bounds to protect against edge interpolations outside original bounding boxes
        elev_tile = np.clip(elev_tile, 0.0, 1.0)
        elev_tile = np.expand_dims(elev_tile, axis=0)  # Shape: [1, H, W]
        
        # read in ground truth mask
        mask_tile = self.mask_src.read(1, window=window).astype(np.int64)
        
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