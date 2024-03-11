import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, root_dir,mask_dir, train_flag):
        """
        Set the root directory , get list of images
        transform to tensor and normalize the data
        """
        print("Initialize Custom Data")
        self.root_dir = root_dir
        self.train_flag = train_flag
        if train_flag:
            self.mask_dir = mask_dir
            self.mask_list = [file for file in os.listdir(mask_dir) if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]

        self.image_list = [file for file in os.listdir(root_dir) if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Adjust normalization as needed
            # mean and standard deviation tuple sent as parameter for normalization
            transforms.Resize((256, 256), antialias=True)
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('L') # convert to gray scale
        
        if self.train_flag:
            mask_name = os.path.join(self.mask_dir, self.image_list[idx])  # Assuming masks have the same filenames
            mask = Image.open(mask_name).convert('L')
            mask = self.transform(mask) 

        image = self.transform(image)
        
        #print("getItem:: image length::",image.shape)
        #print("getItem:: mask length::",mask.shape)

        if self.train_flag:
            return image, mask
        else:
            return image

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
            nn.MaxPool2d(2, 2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        print("x shape::",x.shape)
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


def loadCustomData(img_dir,msk_dir,train_flag=True):

    batchSize = 30
    # get the dataset
    dataset = CustomDataset(root_dir=img_dir, mask_dir=msk_dir, train_flag=train_flag)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=0)

    print("Train flag::", train_flag)
    print("Train loader root directory::",loader.dataset.root_dir)
    print("Total train Size:::",len(loader.dataset.image_list))

    return loader


def runModel(train_loader):

    # Instantiate the U-Net model
    in_channels = 1  # Assuming gray input
    out_channels = 1  # Number of classes for segmentation
    device = torch.device('cuda')
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    #model = UNet(in_channels, out_channels).to(device)
    model = UNet(in_channels, out_channels).to(device)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss for image-to-image translation
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10 # Adjust as needed

    #loss_val = []
    loss_df = pd.DataFrame(columns=['epoch', 'loss_val'])

    for epoch in range(num_epochs):
        #print(f"Run of epoch {epoch} begin..")
        model.train()  # Set the model to training mode
        for images, labels in train_loader:

            #images, labels = images.to(device), labels.to(device)

            # reshape the images and the labels
            images = images.reshape(-1, 1, 256, 256).to(device)
            labels = labels.reshape(-1, 1, 256, 256).to(device) 

            # Forward pass
            outputs = model.forward(images)

            # For image-to-image translation, you may use a different loss function like L1 or L2 loss
            #loss = criterion(outputs, images) # pass the mask with the output as the lables
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        # add the record to the loss df
        loss_df.loc[len(loss_df)] = [epoch+1, loss.item()]
        #loss_val.append(loss.item())

    #plotLoss(loss_df['loss_val'], loss_df['epoch'])
    loss_df.plot(x='epoch', y='loss_val')
    
    # Write the DataFrame to a CSV file
    loss_df.to_csv('../Result/LossOutput.csv', index=False)

def main(img):

    train_dir = train_dir_dict[img]
    mask_dir = train_mask_dir_dict[img]

    #print(f"Inside Main::::{train_dir}")

    # load custom dataset
    train_loader = loadCustomData(train_dir,mask_dir, True)

    # run the model
    runModel(train_loader)

    #test_dir = test_dir_dict[img]
    #mask_dir_test = test_mask_dir_dict[img]

    #test_loader = loadCustomData(test_dir, )

    # run the model


    
if __name__ == '__main__':
    my_path = "/Users/jiten/Masters/WorkPlace/"
    folder_path = os.getcwd() #"/Users/jiten/Masters/WorkPlace/MRI Fractures Project/"

    folder_path = os.path.dirname(folder_path)

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