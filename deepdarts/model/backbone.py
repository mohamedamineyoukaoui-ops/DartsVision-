"""
YOLOv4-tiny Backbone for Keypoint Detection
Simplified implementation based on the DeepDarts paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNLeaky(nn.Module):
    """Convolution + BatchNorm + LeakyReLU block"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Cross Stage Partial Block - key component of YOLOv4"""
    
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        hidden_channels = out_channels // 2
        
        # Main branch
        self.conv1 = ConvBNLeaky(in_channels, hidden_channels, 1)
        
        # Residual branch
        self.conv2 = ConvBNLeaky(in_channels, hidden_channels, 1)
        
        # Residual blocks
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                ConvBNLeaky(hidden_channels, hidden_channels, 1),
                ConvBNLeaky(hidden_channels, hidden_channels, 3, padding=1)
            ) for _ in range(num_blocks)]
        )
        
        # Transition
        self.conv3 = ConvBNLeaky(hidden_channels, hidden_channels, 1)
        self.conv4 = ConvBNLeaky(hidden_channels * 2, out_channels, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.blocks(x2)
        x2 = self.conv3(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv4(x)


class YOLOv4TinyBackbone(nn.Module):
    """
    Simplified YOLOv4-tiny backbone for keypoint detection.
    Returns multi-scale feature maps for detection heads.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initial convolution
        self.conv1 = ConvBNLeaky(3, 32, 3, stride=2, padding=1)  # /2
        
        # Downsampling blocks
        self.conv2 = ConvBNLeaky(32, 64, 3, stride=2, padding=1)  # /4
        self.csp1 = CSPBlock(64, 64, num_blocks=1)
        
        self.conv3 = ConvBNLeaky(64, 128, 3, stride=2, padding=1)  # /8
        self.csp2 = CSPBlock(128, 128, num_blocks=2)
        
        self.conv4 = ConvBNLeaky(128, 256, 3, stride=2, padding=1)  # /16
        self.csp3 = CSPBlock(256, 256, num_blocks=2)
        
        self.conv5 = ConvBNLeaky(256, 512, 3, stride=2, padding=1)  # /32
        self.csp4 = CSPBlock(512, 512, num_blocks=1)
        
    def forward(self, x):
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of feature maps at different scales
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.csp1(x)
        
        x = self.conv3(x)
        x = self.csp2(x)
        feat_small = x  # /8 scale
        
        x = self.conv4(x)
        x = self.csp3(x)
        feat_medium = x  # /16 scale
        
        x = self.conv5(x)
        x = self.csp4(x)
        feat_large = x  # /32 scale
        
        return feat_small, feat_medium, feat_large


class SPPBlock(nn.Module):
    """Spatial Pyramid Pooling block"""
    
    def __init__(self, in_channels, out_channels, pool_sizes=(5, 9, 13)):
        super().__init__()
        self.conv1 = ConvBNLeaky(in_channels, in_channels // 2, 1)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) 
            for k in pool_sizes
        ])
        self.conv2 = ConvBNLeaky(in_channels // 2 * (len(pool_sizes) + 1), out_channels, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        features = [x] + [pool(x) for pool in self.pools]
        x = torch.cat(features, dim=1)
        return self.conv2(x)


if __name__ == "__main__":
    # Test the backbone
    model = YOLOv4TinyBackbone()
    x = torch.randn(1, 3, 416, 416)
    feat_small, feat_medium, feat_large = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Small features (/8): {feat_small.shape}")
    print(f"Medium features (/16): {feat_medium.shape}")
    print(f"Large features (/32): {feat_large.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
