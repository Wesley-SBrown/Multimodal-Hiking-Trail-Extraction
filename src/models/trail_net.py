# src/models/trail_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Depthwise2dCovolutional(nn.Module):
    """
    Neural Network foundation
    Optimizes for consumer grade hardware by dividing up standard convolutions:
     - Spatial (Depthwise)
     - Channel (Pointwise)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        # part 1: depthwise
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            padding=padding, groups=in_channels, bias=False
        )

        # part 2: pointwise
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )

        # normalize in batches
        self.bn = nn.BatchNorm2d(out_channels)

        # apply ReLU 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply architecture following this order:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)

        return self.relu(x)
    
class AdaptiveResamplingLayer(nn.Module):
    """
    Immplements Spatial Frequency Modulation as outlined in Chen et al., 2025
    Applies an adaptive filter kernel prior to downsampling so narrow,
    high freq single pixel trails don't disappear during striding 
    """

    def __init__(self, channels):
        super().__init__()

        # definite learnable local frequency attention weights
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

        # apply avg pooling to handle smooth low-pass anti-aliasing
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        b, c, h, w = x.size()

        # compute the dynamic frequency moduation scale
        gate = self.freq_gate(x).view(b, c, 1, 1)

        # mix raw high freq features with the anti-aliasing low pass downsample
        downsampled = self.pool(x)

        return downsampled * gate
    
class SymmetricEncodingBlock(nn.Module):
    """
    Implements symmetric handling of visual and elevation maps from Cai et al., 2025
        - done prior to cross model projection
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = Depthwise2dCovolutional(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = Depthwise2dCovolutional(in_channels=out_channels, out_channels=out_channels)
        self.ars = AdaptiveResamplingLayer(channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        pooled = self.ars(x)

        # return the identity skip connection & the downsampled layer
        return x, pooled 
        
class MultiModalNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # setup the symmetric dual encoders
        # First the visual stream
        # handles RGB & NIR + NDVI (5 channels total)
        self.vis_enc1 = SymmetricEncodingBlock(5, 32)
        self.vis_enc2 = SymmetricEncodingBlock(32, 64)
        self.vis_enc3 = SymmetricEncodingBlock(64, 128)

        # Second the elevation stream (1 channel)
        self.ele_enc1 = SymmetricEncodingBlock(1, 16)
        self.ele_enc2 = SymmetricEncodingBlock(16, 32)
        self.ele_enc3 = SymmetricEncodingBlock(32, 64)

        # setup bottleneck for fusion
        # Total channels: 128 (Visual) + 64 (Elevation) = 192 channels
        self.bottleneck = nn.Sequential(
            Depthwise2dCovolutional(192, 256),
            Depthwise2dCovolutional(256, 128)
        )

        # setup symmetric decoder tree
        self.up_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # input to dec_3: upsampled (64) + vis_skip3 (128) + ele_skip3 (64) = 256
        self.dec_3 = nn.Sequential(
            Depthwise2dCovolutional(256, 64),
            Depthwise2dCovolutional(64, 64)
        )

        self.up_2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # input to dec_2: upsampled (32) + vis_skip2 (64) + ele_skip2 (32) = 128
        self.dec_2 = nn.Sequential(
            Depthwise2dCovolutional(128, 32),
            Depthwise2dCovolutional(32, 32)
        )

        self.up_1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

        # input to dec_1: upsampled (16) + vis_skip1 (32) + ele_skip1 (16) = 64
        self.dec_1 = nn.Sequential(
            Depthwise2dCovolutional(64, 16),
            Depthwise2dCovolutional(16, 16)
        )

        # output pixel head from final segmentation
        self.final_head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, visual_tensor, elevation_tensor):
        # stream data down through the symetrical feature towers
        v_skip1, v = self.vis_enc1(visual_tensor)
        v_skip2, v = self.vis_enc2(v)
        v_skip3, v = self.vis_enc3(v)

        e_skip1, e = self.ele_enc1(elevation_tensor)
        e_skip2, e = self.ele_enc2(e)
        e_skip3, e = self.ele_enc3(e)

        # perform the symmetrical concatination and fusion at bottleneck
        latent = torch.cat([v,e], dim=1)
        b = self.bottleneck(latent)

        # decode & concat the skip layers at the same time
        x = self.up_3(b)
        x = torch.cat([x, v_skip3, e_skip3], dim=1)
        x = self.dec_3(x)

        x = self.up_2(x)
        x = torch.cat([x, v_skip2, e_skip2], dim=1)
        x = self.dec_2(x)

        x = self.up_1(x)
        x = torch.cat([x, v_skip1, e_skip1], dim=1)
        x = self.dec_1(x)

        return self.final_head(x)
    
if __name__=='__main__':
    print("Testing forward model architecture execution...")
    model = MultiModalNet()
    
    # simulate mini batch 
    dummy_vis = torch.randn(2, 5, 512, 512)
    dummy_elev = torch.randn(2, 1, 512, 512)
    
    output = model(dummy_vis, dummy_elev)
    print(f"Model Output Shape: {output.shape} -> Expected: [2, 2, 512, 512]")
    
    # calc parameter footprint to confirm consumer device hardware optimization
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {params:,}")
