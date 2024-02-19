import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, root_dir,mask_dir):
        """
        Set the root directory , get list of images
        transform to tensor and normalize the data
        """
        print("Initialize Custom Data")
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.image_list = os.listdir(root_dir)
        self.mask_list = os.listdir(mask_dir)
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
        mask_name = os.path.join(self.mask_dir, self.image_list[idx])  # Assuming masks have the same filenames

        image = Image.open(img_name)
        mask = Image.open(mask_name).convert('L')  # Convert to grayscale if needed

        image = self.transform(image)
        mask = self.transform(mask)

        return {'image': image, 'mask': mask}
        #return{'image': image}

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


def loadCustomData(img_dir,msk_dir):

    # get the dataset
    dataset = CustomDataset(root_dir=img_dir, mask_dir=msk_dir)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    print(train_loader.dataset.root_dir)
    print(len(train_loader.dataset.image_list))

    return train_loader


def runModel(train_loader):

    # Instantiate the U-Net model
    in_channels = 1  # Assuming gray input
    out_channels = 1  # Number of classes for segmentation
    model = UNet(in_channels, out_channels)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss for image-to-image translation
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1  # Adjust as needed

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

def main(img):

    train_dir = train_dir_dict[img]
    mask_dir = train_mask_dir_dict[img]

    print(f"Inside Main::::{train_dir}")

    # load custom dataset
    train_loader = loadCustomData(train_dir,mask_dir)

    # run the model
    runModel(train_loader)
    
if __name__ == '__main__':
    my_path = "/Users/jiten/Masters/WorkPlace/"
    folder_path = "/Users/jiten/Masters/WorkPlace/MRI Fractures Project/"

    #source_folder = os.path.join(folder_path, 'SAGT1_Images')
    train_dir_SAGT1 = os.path.join(folder_path, "train_data_SAGT1")
    test_dir_SAGT1 = os.path.join(folder_path, "test_data_SAGT1")
        
    train_dir_dict = {}
    test_dir_dict ={}
    #val_dir = os.path.join(folder_path, "validate")
    train_dir_SAGT1 = os.path.join(folder_path, "train_data_SAGT1")
    test_dir_SAGT1 = os.path.join(folder_path, "test_data_SAGT1")

    train_dir_SAGIR = os.path.join(folder_path, "train_data_SAGIR")
    test_dir_SAGIR = os.path.join(folder_path, "test_data_SAGIR")

    train_dir_dict['SAGT1'] = train_dir_SAGT1
    train_dir_dict['SAGIR'] = train_dir_SAGIR

    test_dir_dict['SAGT1'] = test_dir_SAGT1
    test_dir_dict['SAGIR'] = test_dir_SAGIR

    # create the data structure for mask directories
    train_mask_dir_dict = {}
    test_mask_dir_dict = {}

    train_mask_dir_SAGT1 = os.path.join(folder_path, "train_mask_SAGT1")
    test_mask_dir_SAGT1 = os.path.join(folder_path, "test_mask_SAGT1")

    train_mask_dir_SAGIR = os.path.join(folder_path, "train_mask_SAGIR")
    test_mask_dir_SAGIR = os.path.join(folder_path, "test_mask_SAGIR")

    train_mask_dir_dict['SAGT1'] = train_mask_dir_SAGT1
    train_mask_dir_dict['SAGIR'] = train_mask_dir_SAGIR

    test_mask_dir_dict['SAGT1'] = test_mask_dir_SAGT1
    test_mask_dir_dict['SAGIR'] = test_mask_dir_SAGIR
    print("Call main for SAGIR")
    main('SAGIR')