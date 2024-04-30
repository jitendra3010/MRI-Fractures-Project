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
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from itertools import chain

class Agent:
    def __init__(self, train_flag, img_dir, msk_dir, folder_path, batchSize=10, num_epochs=1, state='new', bilinear=False):
        self.train_flag = train_flag
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.batchSize = batchSize
        self.num_epochs = num_epochs
        self.in_channels = 1    # Assuming gray input
        self.out_channels = 1   # No of classes for segmentation
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        #self.device = torch.device('cuda')
        print(self.device)
        self.model =  None
        self.state = state
        self.bilinear = bilinear
        self.folder_path = folder_path
        self.models_path = os.path.join(folder_path, "Models") 



    
    def initializeUnet(self, file_name=None):
        if self.state == 'new':
            self.model = UNet(self.in_channels, self.out_channels, self.bilinear).to(self.device)
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
        loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=False, num_workers=0)

        print("Train flag::", self.train_flag)
        print("loader root directory::",loader.dataset.root_dir)
        print("Total  Size:::",len(loader.dataset.image_list))

        return loader
    
    def runModel(self, loader):

    
        # Define the loss function and optimizer
        #criterion = nn.CrossEntropyLoss
        criterion = nn.MSELoss()  # Mean Squared Error Loss for image-to-image translation
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

         #loss_val = []
        loss_df = pd.DataFrame(columns=['epoch', 'loss_val'])
        iou_df = pd.DataFrame(columns=['epoch', 'IoU_Score'])

        if self.train_flag:

            for epoch in range(self.num_epochs):
                #print(f"Run of epoch {epoch} begin..")
                prediction_batch = []
                loss_batch = []
                iou_score_batch = []
                self.model.train()  # Set the model to training mode
                for images, labels in loader:

                    # reshape the images and the labels to gray 
                    #images = images.to(self.device) #images.reshape(-1, 1, 256, 256).to(self.device)
                    #labels = labels.to(self.device) #labels.reshape(-1, 1, 256, 256).to(self.device) 
                    images = images.reshape(-1, 1, 256, 256).to(self.device)
                    labels = labels.reshape(-1, 1, 256, 256).to(self.device) 

                    # Forward pass
                    outputs = self.model.forward(images)

                    # Convert predictions to numpy arrays
                    predictions = outputs.detach().cpu().numpy()

                    prediction_batch.append(predictions)
                    # print(predictions.shape)
                    
                    # For image-to-image translation, we may use a different loss function like L1 or L2 loss
                    loss = criterion(outputs, labels)
                    loss_batch.append(loss.item())
                    #print("Loss:", loss)

                    # threshold the outputs before computiing the IUO
                    thresh_image = torch.where(outputs > 0.4, 1, 0)

                    # compute IOU
                    intersection = torch.logical_and(thresh_image, labels).sum().item()
                    union = torch.logical_or(thresh_image, labels).sum().item()

                    # Avoid division by zero
                    if union == 0:
                        iou_score = 0.0
                    else:
                        # Compute IoU
                        iou_score = intersection / union
                    
                    #print(iou_score)
                    iou_score_batch.append(iou_score)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # compute the average loss of the epoch
                avg_Loss_epoch = sum(loss_batch)/ len(loss_batch)
                avg_iou_epoch = sum(iou_score_batch) / len(iou_score_batch)

                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_Loss_epoch}')
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], IoU_score: {avg_iou_epoch}')

                # add the record to the loss df and iou df
                loss_df.loc[len(loss_df)] = [epoch+1, avg_Loss_epoch]
                iou_df.loc[(len(iou_df))] = [epoch +1, avg_iou_epoch]

            return loss_df, prediction_batch, iou_df
        
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
            predictions = torch.cat(predictions, dim=0)
            return predictions


    def writeRun(self, dataframe, filename):
        '''Function to write the data to the file'''
        dataframe.to_csv(filename, index=False)

    def savePredictions(self, loader, predictions):

        image_names = loader.dataset.image_list

        pred_save_path = os.path.join(self.folder_path,"Predictions")

        for name, pred in zip(image_names, predictions):
            file_path = os.path.join(pred_save_path,name)

            # Convert the tensor to a numpy
            img_pil = pred.cpu().detach().numpy()

            predicted_image = Image.fromarray(img_pil.astype(np.uint8))

            predicted_image.save(file_path)

    def printPrediction(self, loader, preds):
        '''function to print few predictions with image and mask'''
        
        img_names = loader.dataset.image_list

        # no of images you want to show
        counter = 3 

        image_names = img_names[counter:counter+3]
        pred_image = preds[counter:counter+3]


        #pred_save_path = os.path.join(self.folder_path,"Predictions")

        for name, predictions in zip(image_names, pred_image): # TO:DO uncomment this loop later
            print(name)
            #file_path = os.path.join(pred_save_path,name)
            image = os.path.join(self.img_dir, name)
            img = Image.open(image).convert('L')

            mask = os.path.join(self.msk_dir, name)
            msk = Image.open(mask).convert('L')

            
            #print(f"Max :: {np.max(predictions)}")
            #print(f"Min :: {np.min(predictions)}")

            # threshold to 0 or 1 based on mean pixel value
            thresholded_image = np.where(predictions > 0.4, 1, 0)

            #print(f"Max :: {np.max(thresholded_image)}")
            #print(f"Min :: {np.min(thresholded_image)}")

            #print(thresholded_image.squeeze().shape)
            #print(thresholded_image.squeeze())

            # Create a figure and axis objects
            fig, axs = plt.subplots(1, 4, figsize=(15, 5))

            # Plot the images
            axs[0].imshow(img, cmap='gray')
            axs[0].set_title('Original Image')

            axs[1].imshow(msk, cmap='gray')
            axs[1].set_title('Mask Image')

            axs[2].imshow(predictions.squeeze(), cmap='gray')
            axs[2].set_title('Prediction Image')

            axs[3].imshow(thresholded_image.squeeze(), cmap='gray')
            axs[3].set_title('Threshold Image')

            # Hide the axis
            for ax in axs:
                ax.axis('off')

            # Display the plot
            plt.show()


    # def computeIoU(self, loader, preds):
    #     '''Function to compute the IoU of the prediction to ground truth'''

    #     # get the image names from the loader
    #     img_names = loader.dataset.image_list
        
    #     iou_score_all = []
    #     #iou_df = pd.DataFrame(columns=['ImageNo', 'IoU_Score'])

    #     # loop through the names and prediciton to compute the 
    #     for name, predictions in zip(img_names, preds):
    #         mask = os.path.join(self.msk_dir, name)
    #         msk = Image.open(mask).convert('L')

    #         # threshold to 0 or 1 based on mean pixel value
    #         thresh_image = np.where(predictions > 0.4, 1, 0)

    #         # append the iou Score to the list
    #         thresh_image = thresh_image.squeeze().astype(np.uint8)

    #         iou_score = self.iou_score(np.array(thresh_image), msk)
    #         #iou_df.loc[len(iou_df)] = [len(iou_df)+1, iou_score]
    #         iou_score_all.append(iou_score)

    #     iou_sore_avg = sum(iou_score_all) / len(iou_score_all)

    #     return iou_sore_avg

    # def iou_score(self, pred_mask, true_mask):
    #     '''Fuction to compute the mask'''
        
    #     # Resize the predicted mask to match the dimensions of the true mask
    #     pred_mask = Image.fromarray(pred_mask)
    #     pred_mask_resized = pred_mask.resize(true_mask.size, Image.NEAREST)

    #     intersection = np.logical_and(pred_mask_resized, true_mask)
    #     union = np.logical_or(pred_mask_resized, true_mask)
    #     iou_score = np.sum(intersection) / np.sum(union)

    #     return iou_score

