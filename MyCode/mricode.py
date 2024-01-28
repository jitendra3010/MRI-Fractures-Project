import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

#/content/train_data/P001 SAGT1_004.jpg

#my_path = "/Users/Jiten/Masters/WorkPlace/MRI Fractures Project/SAGTImages"
train_path = "/content/train_data"
test_path = "/content/test_data"

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        """
        Set the root directory , get list of images
        transform to tensor and normalize the data
        """
        self.root_dir = root_dir
        self.image_list = os.listdir(root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Adjust normalization as needed
            # mean and standard deviation tuple sent as parameter for normalization
            transforms.Resize((640, 640))
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        #mask_name = os.path.join(self.root_dir, 'masks', self.image_list[idx])  # Assuming masks have the same filenames

        image = Image.open(img_name)
        #mask = Image.open(mask_name).convert('L')  # Convert to grayscale if needed

        image = self.transform(image)
        #mask = self.transform(mask)

        #return {'image': image, 'mask': mask}
        return{'image': image}

# unet architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting path
        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expansive path
        self.upconv4 = self.upconv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
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
        upconv1 = torch.cat([upconv1, conv1], dim=1)

        # Output layer
        output = self.out_conv(upconv1)

        return output

# def collate_fn(batch):
#   images = [resize_image(image, size=(640, 640)) for image in batch['image']]
#   return torch.stack(images, dim=0)

# get the dataset
dataset = CustomDataset(root_dir=train_path)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

print(train_loader.dataset.root_dir)
print(len(train_loader.dataset.image_list))

# Instantiate the U-Net model
in_channels = 1  # Assuming gray input
out_channels = 1  # Number of classes for segmentation
model = UNet(in_channels, out_channels)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for image-to-image translation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch in train_loader:
        images = batch['image']

        images = images.reshape(-1, 1, 640, 640)

        # Forward pass
        outputs = model(images)

        # For image-to-image translation, you may use a different loss function like L1 or L2 loss
        loss = criterion(outputs, images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')