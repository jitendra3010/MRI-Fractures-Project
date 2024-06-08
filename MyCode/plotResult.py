import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotLoss_IOU(lossDf, iouDf):
    """
    Plot the x an y
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(lossDf['epoch'], lossDf['loss_val'])
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss in each epoch")
    ax[0].grid(True)

    ax[1].plot(iouDf['epoch'], iouDf['IoU_Score'])
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Iou_Score")
    ax[1].set_title("IouScore ")
    ax[1].grid(True)
    
    plt.show()

def plotTest_IOU(iou_test):
    # fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    # Plot histogram
    plt.hist(iou_test, bins=30, color='blue', alpha=0.7)

    # Set labels and title
    plt.xlabel("Iou_score")
    plt.ylabel("Frequency")
    plt.title("Histogram of Iou Score Test")
    plt.show()


def plot_img_and_mask(img, mask):

    # Print predictions
    #fig, ax = plt.subplots(1, 2)
    plt.imshow(mask.squeeze(), cmap='gray')  # Assuming grayscale predictions
    plt.title('Model Prediction')
    plt.colorbar()
    plt.show()

    plt.imshow(img.squeeze(), cmap='gray')  # Assuming grayscale predictions
    plt.title('Model image')
    plt.colorbar()
    plt.show()
    
    # classes = mask.max() + 1
    # fig, ax = plt.subplots(1, classes + 1)
    # ax[0].set_title('Input image')
    # ax[0].imshow(img)
    # for i in range(classes):
    #     ax[i + 1].set_title(f'Mask (class {i + 1})')
    #     ax[i + 1].imshow(mask == i)
    # plt.xticks([]), plt.yticks([])
    # plt.show()

def plotOptimalThresh(iou_vs_thresh):
    '''
    Plot the differnet iou score vs threshold
    '''
    threshold = np.arange(0.1, 1.1, 0.1)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(threshold, iou_vs_thresh, marker='o')

    # Add labels and title
    plt.title('IoU Score vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IoU Score')

    # Show grid
    plt.grid(True)

    # Show plot
    plt.show()

def main():

    lossDf = pd.read_csv(lossFile)
    iouDf = pd.read_csv(iouFile)
    iouDf_test = pd.read_csv(iouTestfile)

    plotLoss_IOU(lossDf, iouDf)

    plotTest_IOU(iouDf_test)

if __name__ == '__main__':

    folder_path = os.getcwd() 

    lossFile = os.path.join(folder_path, 'Result', "LossOutput.csv")
    iouFile = os.path.join(folder_path, 'Result', "IoUScore.csv")
    iouTestfile = os.path.join(folder_path, 'Result', "IoUScore_test.csv")
    
    main()