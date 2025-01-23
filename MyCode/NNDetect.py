import torch
import torch.nn as nn
import torch.nn.functional as F

class NNDetect(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(NNDetect, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(input_channels, 8)    # Start with 8 filters
        self.enc2 = self.conv_block(8, 16)                # Double filters
        self.enc3 = self.conv_block(16, 32)
        self.enc4 = self.conv_block(32, 64)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)        # Maximum filters at 128

        # Decoder with ConvTranspose2d for upsampling
        self.dec4 = self.upconv_block(128 + 64, 64)       # Concatenate, reduce filters
        self.dec3 = self.upconv_block(64 + 32, 32)
        self.dec2 = self.upconv_block(32 + 16, 16)
        self.dec1 = self.upconv_block(16 + 8, 8)

        # Output
        self.output = nn.Conv2d(8, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        print("upconv_block::",in_channels, out_channels)
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [batch_size, 8, 256, 256]
        #print("e1 shape::", e1.shape)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))  # [batch_size, 16, 128, 128]
        #print("e2 shape::", e2.shape)
        e3 = self.enc3(nn.MaxPool2d(2)(e2))  # [batch_size, 32, 64, 64]
        #print("e3 shape::", e3.shape)
        e4 = self.enc4(nn.MaxPool2d(2)(e3))  # [batch_size, 64, 32, 32]
        #print("e4 shape::", e4.shape)

        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))  # [batch_size, 128, 16, 16]
        #print("b shape::", b.shape)

        # Upsample b to match e4 if required
        if b.size(2) != e4.size(2) or b.size(3) != e4.size(3):
            b = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True)
        
        #print("aflter interplolation b shape::", b.shape)

        # Decoder with ConvTranspose2d for upsampling
        d4 = self.dec4(torch.cat([b, e4], dim=1))  # [batch_size, 64, 32, 32]
        #print("d4 shape::", d4.shape)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # [batch_size, 32, 64, 64]
        #print("d3 shape::", d3.shape)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # [batch_size, 16, 128, 128]
        #print("d2 shape::", d2.shape)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # [batch_size, 8, 256, 256]
        #print("d1 shape::", d1.shape)

        d1 = d1[:, :, :256, :256]

        # Output layer
        out = self.output(d1)  # [batch_size, 1, 256, 256]
        #print("out shape::", out.shape)
        return torch.sigmoid(out)
