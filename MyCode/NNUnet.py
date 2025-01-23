import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Convolution Block used in nnU-Net
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

# nnU-Net Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        skip_connection = x  # Save the feature map for skip connection
        x = self.pool(x)
        return x, skip_connection

# nnU-Net Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        #print("inside Init",in_channels, out_channels)

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x, skip_connection):
        #print("Inside forward",x.shape, skip_connection.shape)
        x = self.upconv(x)

        #print("after upconv",x.shape, skip_connection.shape)
        # Match spatial dimensions of skip connection and upsampled output
        if x.size(2) != skip_connection.size(2) or x.size(3) != skip_connection.size(3):
            x = F.interpolate(x, size=(skip_connection.size(2), skip_connection.size(3)), mode='bilinear', align_corners=True)

        #print("After interpolating",x.shape, skip_connection.shape)
        x = torch.cat([x, skip_connection], dim=1)  # Concatenate along channel axis
        #print("after cat::",x.shape)
        x = self.block(x)
        #print(x.shape)
        return x

# nnU-Net Model (simplified version)
class nnUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(nnUNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, 8)
        self.encoder2 = EncoderBlock(8, 16)
        self.encoder3 = EncoderBlock(16, 32)
        self.encoder4 = EncoderBlock(32, 64)

        self.bottleneck = ConvBlock(64, 128)

        self.decoder4 = DecoderBlock(128, 64)
        self.decoder3 = DecoderBlock(64, 32)
        self.decoder2 = DecoderBlock(32, 16)
        self.decoder1 = DecoderBlock(16, 8)

        self.final_conv = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)

        #print("before bottle neck")
        # Bottleneck
        x = self.bottleneck(x4)
        #print("after bottle neck")

        #print(x.shape, skip4.shape)
        # Decoder
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        # Final convolution to get the segmentation map
        x = self.final_conv(x)
        return x