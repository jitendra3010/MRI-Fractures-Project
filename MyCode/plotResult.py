import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# def plotLoss_IOU(lossDf, iouDf):
#     """
#     Plot the x an y
#     """
#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     ax[0].plot(lossDf['epoch'], lossDf['loss_val'])
#     ax[0].set_xlabel("Epochs")
#     ax[0].set_ylabel("Loss")
#     ax[0].set_title("Loss in each epoch")
#     ax[0].grid(True)

#     ax[1].plot(iouDf['epoch'], iouDf['IoU_Score'])
#     ax[1].set_xlabel("Epochs")
#     ax[1].set_ylabel("Iou_Score")
#     ax[1].set_title("IouScore ")
#     ax[1].grid(True)
    
#     plt.show()

def plotLoss_IOU(train_loss_df, train_iou_df, val_loss_df=None, val_iou_df=None):
    """
    Plot the training and validation loss and IoU score over epochs.

    Parameters:
    - train_loss_df: DataFrame containing 'epoch' and 'loss_val' columns for training data.
    - train_iou_df: DataFrame containing 'epoch' and 'IoU_Score' columns for training data.
    - val_loss_df: DataFrame containing 'epoch' and 'loss_val' columns for validation data (optional).
    - val_iou_df: DataFrame containing 'epoch' and 'IoU_Score' columns for validation data (optional).
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Training Loss
    ax[0].plot(train_loss_df['epoch'], train_loss_df['loss_val'], label='Train Loss')
    
    # Plot Validation Loss if provided
    if val_loss_df is not None:
        ax[0].plot(val_loss_df['epoch'], val_loss_df['loss_val'], label='Val Loss', linestyle='--')
    
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss over Epochs")
    ax[0].grid(True)
    ax[0].legend()
    
    # Plot Training IoU Score
    ax[1].plot(train_iou_df['epoch'], train_iou_df['IoU_Score'], label='Train IoU Score', color='blue')
    
    # Plot Validation IoU Score if provided
    if val_iou_df is not None:
        ax[1].plot(val_iou_df['epoch'], val_iou_df['IoU_Score'], label='Val IoU Score', color='orange', linestyle='--')
    
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("IoU Score")
    ax[1].set_title("IoU Score over Epochs")
    ax[1].grid(True)
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()


# def plotTest_IOU(iou_test):
#     # fig, ax = plt.subplots(1, 1, figsize=(15, 5))
#     # Plot histogram
#     plt.hist(iou_test, bins=30, color='blue', alpha=0.7)

#     # Set labels and title
#     plt.xlabel("Iou_score")
#     plt.ylabel("Frequency")
#     plt.title("Histogram of Iou Score Test")
#     plt.show()
def plotTest_IOU(iou_test):
    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    # Plot histogram
    ax.hist(iou_test, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add vertical line for mean
    mean_value = sum(iou_test) / len(iou_test)
    ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    ax.text(mean_value, plt.gca().get_ylim()[1]*0.9, f'Mean: {mean_value:.2f}', color = 'red')

    # Add vertical line for median
    median_value = sorted(iou_test)[len(iou_test) // 2]
    ax.axvline(median_value, color='green', linestyle='dashed', linewidth=1)
    ax.text(median_value, plt.gca().get_ylim()[1]*0.8, f'Median: {median_value:.2f}', color = 'green')

    # Set labels and title
    ax.set_xlabel("IOU Score", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_title("Histogram of IOU Score Test", fontsize=16, fontweight='bold')

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Show plot
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
    #iouDf_test = pd.read_csv(iouTestfile)

    plotLoss_IOU(lossDf, iouDf)

    #plotTest_IOU(iouDf_test)

if __name__ == '__main__':

    folder_path = os.getcwd() 

    lossFile = os.path.join(folder_path, 'Result', "LossOutput.csv")
    iouFile = os.path.join(folder_path, 'Result', "IoUScore.csv")
    #iouTestfile = os.path.join(folder_path, 'Result', "IoUScore_test.csv")
    
    main()