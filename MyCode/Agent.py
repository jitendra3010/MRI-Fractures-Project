from CustomDataset import CustomDataset
from UNet import UNet
from torch.utils.data import Dataset, DataLoader
from torch import save
from torch import load
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import datetime

class Agent:
    def __init__(self, train_flag, img_dir, msk_dir, batchSize=10, num_epochs=1, state='new'):
        self.train_flag = train_flag
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.batchSize = batchSize
        self.num_epochs = num_epochs
        self.in_channels = 1    # Assuming gray input
        self.out_channels = 1   # No of classes for segmentation
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(self.device)
        self.model =  None
        self.state = state
        folder_path = os.getcwd()
        self.models_path = os.path.join(folder_path, "Models") 



    
    def initializeUnet(self, file_name=None):
        if self.state == 'new':
            self.model = UNet(self.in_channels, self.out_channels).to(self.device)
        else:
            self.train_flag = False
            open_path = os.path.join(self.models_path, file_name)
            self.model = load(open_path)

    def save_net(self,file_name):
        save_path = os.path.join(self.models_path, file_name + str(datetime.datetime.now().strftime("%b %d, %Y %I_%M%p")))
        save(self.model, save_path)


    def loadCustomData(self):

        # get the dataset
        dataset = CustomDataset(root_dir=self.img_dir, mask_dir=self.msk_dir, train_flag=self.train_flag)
        loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True, num_workers=0)

        print("Train flag::", self.train_flag)
        print("loader root directory::",loader.dataset.root_dir)
        print("Total  Size:::",len(loader.dataset.image_list))

        return loader
    
    def runModel(self, loader):

    
        # Define the loss function and optimizer
        criterion = nn.MSELoss()  # Mean Squared Error Loss for image-to-image translation
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

         #loss_val = []
        loss_df = pd.DataFrame(columns=['epoch', 'loss_val'])

        if self.train_flag:

            for epoch in range(self.num_epochs):
                #print(f"Run of epoch {epoch} begin..")
                self.model.train()  # Set the model to training mode
                for images, labels in loader:

                    #images, labels = images.to(device), labels.to(device)

                    # reshape the images and the labels
                    images = images.reshape(-1, 1, 256, 256).to(self.device)
                    labels = labels.reshape(-1, 1, 256, 256).to(self.device) 

                    # Forward pass
                    outputs = self.model.forward(images)

                    # For image-to-image translation, you may use a different loss function like L1 or L2 loss
                    #loss = criterion(outputs, images) # pass the mask with the output as the lables
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item()}')

                # add the record to the loss df
                loss_df.loc[len(loss_df)] = [epoch+1, loss.item()]

            # Write the DataFrame to a CSV file
            #loss_df.to_csv('../Result/LossOutput.csv', index=False)
            return loss_df
        
        else:
            predictions = []

            # set the model to evaluation mode
            self.model.eval()

            for images in loader:
                images = images.reshape(-1, 1, 256, 256).to(self.device)

                outputs = self.model.forward(images)

                # Collect predictions
                predictions.append(outputs)

            # Combine predictions from all batches
            #predictions = torch.cat(predictions, dim=0)
            return predictions


    def writeRun(self, dataframe, filename):
        
        dataframe.to_csv(filename, index=False)