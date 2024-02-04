import pandas as pd
import os
from PIL import Image
import cv2

def main():

    # get the bounding info into a dataframe
    sagT1_bounding = pd.read_csv(csv_path)

    createMask(sagT1_bounding, train_dir, train_mask_dir)

if __name__ == '__main__':
    my_path = "/Users/jiten/Masters/WorkPlace/"
    folder_path = os.path.join(my_path,'MRI Fractures Project')

    csv_path = os.path.join(folder_path,'boundingInfo','SAGT1 bounding boxes.csv')
    source_folder = os.path.join(folder_path, 'SAGT1_Images')
    train_dir = os.path.join(folder_path, "train_data")
    #val_dir = os.path.join(folder_path, "validate")
    test_dir = os.path.join(folder_path, "test_data")

    train_mask_dir = os.path.join(folder_path, "train_mask")
    test_mask_dir = os.path.join(folder_path, "test_mask")
    main()


def createMask(sagT1_bounding, source, dest):

    # Get the list of all files in the folder
    all_files = os.listdir(source)

    for file in all_files:
        x,y





