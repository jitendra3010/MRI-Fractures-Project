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

import importlib
import CustomDataset
import EarlyStopping
importlib.reload(CustomDataset)


class DiceLoss(nn.Module):
    '''Loss function for criterion image segmentation'''
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Flatten the tensors to ensure they are 1D
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return 1 - dice

class Agent:
    def __init__(self, train_flag, img_dir, msk_dir, folder_path, val_dir=None, msk_dir_val=None, batchSize=10, num_epochs=1, state='new', bilinear=False):
        self.train_flag = train_flag
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.val_dir = val_dir
        self.msk_dir_val = msk_dir_val
        self.batchSize = batchSize
        self.num_epochs = num_epochs
        self.in_channels = 1    # Assuming gray input
        self.out_channels = 1   # No of classes for segmentation
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(self.device)
        self.model =  None
        self.state = state
        self.bilinear = bilinear
        self.folder_path = folder_path
        self.models_path = os.path.join(folder_path, "Models")
        self.threshold = 0.3 



    
    def initializeUnet(self, file_name=None):
        '''Funciton to initialize the u net model'''

        if self.state == 'new':
            self.model = UNet(self.in_channels, self.out_channels, self.bilinear).to(self.device)
        else:
            self.train_flag = False
            open_path = os.path.join(self.models_path, file_name)
            self.model = load(open_path)

    def save_net(self,file_name):
        '''Function to save the model'''

        save_path = os.path.join(self.models_path, file_name + str(datetime.datetime.now().strftime("%b %d, %Y %I_%M%p")))
        save(self.model, save_path)


    def loadCustomData(self, augment=False):
        '''Function to load the custom data for the model'''

        # get the dataset
        dataset = CustomDataset.CustomDataset(root_dir=self.img_dir, mask_dir=self.msk_dir, train_flag=self.train_flag, augment=augment)
        loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=False, num_workers=0)

        print("Augmentation :::", augment)
        print("Train flag::", self.train_flag)
        print("loader root directory::",loader.dataset.root_dir)
        print("Total Size :::",len(dataset))

        if(self.train_flag):
            val_dataset = CustomDataset.CustomDataset(root_dir=self.val_dir, mask_dir=self.msk_dir_val, train_flag=self.train_flag)
            val_loader = DataLoader(val_dataset, batch_size=self.batchSize, shuffle=False, num_workers=0)
            print("Total Size val:::",len(val_dataset))
            return loader, val_loader

        return loader
    
    def runModel(self, loader, val_loader=None, earlyStop=False, L2Reg=False, loss='MSE'):
        '''Function to run the model for train and test based on trainflag'''

        if earlyStop:
            early_stopping = EarlyStopping.EarlyStopping(patience=5, verbose=True)
    
        # Define the loss function and optimizer
        #criterion = nn.CrossEntropyLoss
        if loss == 'MSE':
            criterion = nn.MSELoss()  # Mean Squared Error Loss for image-to-image translation
        elif loss == 'DICE':
            criterion = DiceLoss() # Dice loss to segmentation
        
        # if no L2 regularization
        if not L2Reg:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            # Adding L2 regularization (weighted Decay) to combat overfitting by introducing a penalty term
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)


         #loss_val = []
        loss_df = pd.DataFrame(columns=['epoch', 'loss_val'])
        iou_df = pd.DataFrame(columns=['epoch', 'IoU_Score'])

        loss_val_df = pd.DataFrame(columns=['epoch', 'loss_val'])
        iou_val_df = pd.DataFrame(columns=['epoch', 'IoU_Score'])

        if self.train_flag:

            for epoch in range(self.num_epochs):
                #print(f"Run of epoch {epoch} begin..")
                prediction_batch = []
                loss_batch = []
                iou_score_batch = []

                # initialize for early stop
                #prev_iou_score = 0

                ###################### Train Mode ##############################

                self.model.train()  # Set the model to training mode
                for images, labels in loader:

                    # reshape the images and the labels
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

                    # get the iou_score of the batch of loader
                    iou_score = self.getIoUBatch(outputs, labels)
                    
                    #print(iou_score)
                    iou_score_batch.append(iou_score)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # compute the average loss of the epoch
                avg_Loss_epoch = sum(loss_batch)/ len(loss_batch)
                avg_iou_epoch = sum(iou_score_batch) / len(iou_score_batch)

                #print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_Loss_epoch}')
                #print(f'Epoch [{epoch + 1}/{self.num_epochs}], IoU_score: {avg_iou_epoch}')

                # add the record to the loss df and iou df
                loss_df.loc[len(loss_df)] = [epoch+1, avg_Loss_epoch]
                iou_df.loc[(len(iou_df))] = [epoch +1, avg_iou_epoch]

                # do an early stop if the change in iou score is minimal
                #if abs(prev_iou_score - avg_iou_epoch) < 0.001:
                #    break

                # validaiton run
                val_loss_batch = []
                iou_score_val_batch = []
                prediction_batch_val = []

                ##############################################################################

                ############################# Validation mode ################################

                self.model.eval() # set the model to evaluation mode
                with torch.no_grad():
                    for images_val, labels_val in val_loader:
                        # reshape the images and the labels
                        images_val = images_val.reshape(-1, 1, 256, 256).to(self.device)
                        labels_val = labels_val.reshape(-1, 1, 256, 256).to(self.device) 

                        # Forward pass
                        outputs_val = self.model.forward(images_val)

                        # Convert predictions to numpy arrays
                        predictions_val = outputs_val.detach().cpu().numpy()

                        # Collect predictions
                        prediction_batch_val.append(predictions_val)

                        # get the loss
                        loss_val = criterion(outputs_val, labels_val)

                        val_loss_batch.append(loss_val.item())

                        # get the iou_score of the batch of loader
                        iou_score = self.getIoUBatch(outputs_val, labels_val)

                        iou_score_val_batch.append(iou_score)

                ##############################################################################

                # compute the average validation loss of the epoch
                avg_Loss_val_epoch = sum(val_loss_batch)/ len(val_loss_batch)
                avg_iou_val_epoch = sum(iou_score_val_batch) / len(iou_score_val_batch)

                 # add the record to the loss df and iou df
                loss_val_df.loc[len(loss_df)] = [epoch+1, avg_Loss_val_epoch]
                iou_val_df.loc[(len(iou_df))] = [epoch +1, avg_iou_val_epoch]

                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train( Loss: {avg_Loss_epoch:.4f}, IoU_score: {avg_iou_epoch:.4f} )'
                      f' ::: Validation ( Loss: {avg_Loss_val_epoch:.4f}, IoU_score: {avg_iou_val_epoch:.4f} )')
                #print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train IoU_score: {avg_iou_epoch}, Validaiton IoU_score: {avg_iou_val_epoch}')

                # check for ealry stopping
                if earlyStop:
                    early_stopping(avg_Loss_val_epoch, self.model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

            # no requirement to send the validation information back
            return loss_df, prediction_batch, iou_df, loss_val_df, iou_val_df, prediction_batch_val
        
        else:
            ########################### Model Testing ####################################
            predictions = []

            # set the model to evaluation mode
            self.model.eval()

            # dataframe to capture the prediciotn and iou score
            prediction_batch = []
            iou_score_batch = []

            for images, labels in loader:
                
                # reshape the image and labels
                images = images.reshape(-1, 1, 256, 256).to(self.device)
                labels = labels.reshape(-1, 1, 256, 256).to(self.device)
                #print("len of images::",len(images))
                #print("len of labels::",len(labels))

                # get the output
                outputs = self.model.forward(images)
                #print(len(outputs))

                # Convert predictions to numpy arrays
                predictions = outputs.detach().cpu().numpy()

                # Collect predictions
                prediction_batch.append(predictions)

                # get the iou_score of the batch of loader
                iou_score = self.getIoUBatch(outputs, labels)
                    
                print(iou_score)
                # append the iou_score
                iou_score_batch.append(iou_score)
                #print(len(iou_score_batch))

            avg_iou_batch = sum(iou_score_batch) / len(iou_score_batch)
            #print(len(iou_score_batch))
            print(f"Iou Score ::::{avg_iou_batch}")

            # compute iou score of each test image to a list
            # since we are running for 1 epoch with all the test images, the outputs and labels will have all of them
            iou_score_each = self.computeEachIouTest(outputs, labels)

            # Combine predictions from all batches
            #predictions = torch.cat(predictions, dim=0)
            return prediction_batch, iou_score_batch, iou_score_each


    def getIoUBatch(self,outputs, labels):
        '''Get Iou For each batch of images'''

         # threshold the outputs before computiing the IUO
        thresh_image = torch.where(outputs > self.threshold, 1, 0)

        #print(len(thresh_image))

        # compute IOU
        intersection = torch.logical_and(thresh_image, labels).sum().item()
        union = torch.logical_or(thresh_image, labels).sum().item()


        # Avoid division by zero
        if union == 0:
            iou_score = 0.0
        else:
            # Compute IoU
            iou_score = intersection / union

        return iou_score

    def computeEachIouTest(self, outputs, labels):
        
        iou_scores = []

        for output, label in zip(outputs, labels):
            # Threshold the outputs
            thresh_image = torch.where(output > self.threshold, 1, 0)

            # Compute intersection and union
            intersection = torch.logical_and(thresh_image, label).sum().item()
            union = torch.logical_or(thresh_image, label).sum().item()

            # Avoid division by zero
            if union == 0:
                iou_score = 0.0
            else:
                # Compute IoU
                iou_score = intersection / union

            # Append the IoU score to the list of scores
            iou_scores.append(iou_score)
        
        return iou_scores

    def writeRun(self, dataframe, filename):
        '''Function to write the data to the file'''
        dataframe.to_csv(filename, index=False)

    def savePredictions(self, loader, predictions):
        '''Function to save the predictions'''

        image_names = loader.dataset.image_list

        pred_save_path = os.path.join(self.folder_path,"Predictions")

        for name, pred in zip(image_names, predictions):
            file_path = os.path.join(pred_save_path,name)

            # Convert the tensor to a numpy
            img_pil = pred.cpu().detach().numpy()

            predicted_image = Image.fromarray(img_pil.astype(np.uint8))

            predicted_image.save(file_path)

    def printPrediction(self, loader, preds, idx=6, validation=False):
        '''function to print few predictions with image and mask'''

        if not validation:
            imgDir = self.img_dir
            mskDir = self.msk_dir
        else:
            imgDir = self.val_dir
            mskDir = self.msk_dir_val
        
        img_names = loader.dataset.image_list

        # starting index counter
        counter = idx

        image_names = img_names[counter:counter+10]
        pred_image = preds[counter:counter+10]


        #pred_save_path = os.path.join(self.folder_path,"Predictions")

        for name, predictions in zip(image_names, pred_image): 
            print(name)
            #file_path = os.path.join(pred_save_path,name)
            image = os.path.join(imgDir, name)
            img = Image.open(image).convert('L')

            mask = os.path.join(mskDir, name)
            msk = Image.open(mask).convert('L')


            # threshold to 0 or 1 based on mean pixel value
            thresholded_image = np.where(predictions > self.threshold, 1, 0)

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


    def optimalThresVsIoU(self, loader, preds):
        '''
        Function to compute the IoU of the prediction to ground truth for different threhold
        '''

        threshold = np.arange(0.1, 1.1, 0.1)

        # get the image names from the loader
        img_names = loader.dataset.image_list
        
        iou_vs_thresh = []
        #iou_df = pd.DataFrame(columns=['ImageNo', 'IoU_Score'])

        # loop thorugh all the threshold and compute the iou score
        for thresh in threshold:

            iou_score_all = []
            # loop through the names and prediciton to compute the 
            for name, predictions in zip(img_names, preds):
            
                mask = os.path.join(self.msk_dir, name)
                msk = Image.open(mask).convert('L')

                # threshold to 0 or 1 based on mean pixel value
                thresh_image = np.where(predictions > thresh, 1, 0)

                # append the iou Score to the list
                thresh_image = thresh_image.squeeze().astype(np.uint8)

                # ge the iou score of each image
                iou_score = self.iou_score(np.array(thresh_image), msk)

                #iou_df.loc[len(iou_df)] = [len(iou_df)+1, iou_score]
                iou_score_all.append(iou_score)

            iou_sore_avg = sum(iou_score_all) / len(iou_score_all)
            iou_vs_thresh.append(iou_sore_avg)

        return iou_vs_thresh

    def iou_score(self, pred_mask, true_mask):
        '''Fuction to compute the mask of each prediciotn of and true mask'''
        
        # Resize the predicted mask to match the dimensions of the true mask
        pred_mask = Image.fromarray(pred_mask)
        pred_mask_resized = pred_mask.resize(true_mask.size, Image.NEAREST)

        intersection = np.logical_and(pred_mask_resized, true_mask)
        union = np.logical_or(pred_mask_resized, true_mask)
        iou_score = np.sum(intersection) / np.sum(union)

        return iou_score
    
    def computePredictionScore(self, loader, preds):
        '''Function to compute the prediction scores'''

        cols = ['ImageName', 'Image', 'TrueMask', 'PredMask', 'PredScore', 'ActualScore']
        df = pd.DataFrame(columns=cols)

        imgDir = self.img_dir
        mskDir = self.msk_dir

        img_names = loader.dataset.image_list

        for name, prediction in zip(img_names, preds):

            image = os.path.join(imgDir, name)
            img = Image.open(image).convert('L')

            mask = os.path.join(mskDir, name)
            msk = Image.open(mask).convert('L')

             # reshape the image and labels
            img = img.resize((256, 256))
            msk = msk.resize((256,256))

            #print(img.size, msk.size, prediction.)
            # Compute the sum of pixel values for both images
            sum_original = np.sum(np.array(img) > 10)
            sum_predicted = np.sum(prediction.squeeze() > self.threshold)
            sum_msk = np.sum(np.array(msk) == 255)

            # Compute the prediction score
            prediction_score = sum_predicted / sum_original

            # compute the actual score
            actual_score =  sum_msk / sum_original

            df.loc[len(df)] = [name, np.array(img), np.array(msk), np.array(prediction.squeeze()), prediction_score, actual_score]

        
        return df

