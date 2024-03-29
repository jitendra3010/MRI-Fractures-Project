import torch.nn as nn
import torch

# unet architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting path
        self.conv1 = self.conv_block(in_channels, 8)
        self.conv2 = self.conv_block(8, 16 )
        self.conv3 = self.conv_block(16, 32)
        self.conv4 = self.conv_block(32, 64)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Expansive path
        self.upconv4 = self.upconv_block(128, 64)
        self.upconv3 = self.upconv_block(128, 32)
        self.upconv2 = self.upconv_block(64, 48)
        self.upconv1 = self.upconv_block(64, 8)

        # Output layer
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            #nn.MaxPool2d(2, 2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        #print("x shape::",x.shape)
        # Contracting path
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # Bottleneck
        bottleneck = self.bottleneck(conv4)

        # Expansive path
        upconv4 = self.upconv4(bottleneck)
        upconv4 = torch.cat([upconv4, conv4], dim=1)
        upconv3 = self.upconv3(upconv4)
        upconv3 = torch.cat([upconv3, conv3], dim=1)
        upconv2 = self.upconv2(upconv3)
        upconv2 = torch.cat([upconv2, conv2], dim=1)
        upconv1 = self.upconv1(upconv2)
        #print("Conv1::",conv1.shape)
        #print("Before cat::",upconv1.shape)
        upconv1 = torch.cat([upconv1, conv1], dim=1)
        #print("After cat:::",upconv1.shape)
        

        # Output layer
        output = self.out_conv(upconv1)

        return output