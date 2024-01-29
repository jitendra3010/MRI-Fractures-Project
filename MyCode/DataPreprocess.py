import pandas as pd
import os
from PIL import Image
import shutil
import random




def prepareData(imgName):
    """
    Function to copy the images from main data folder to
    a different folder for all SAGT1 images
    """

    print("Start of Copy files")
    counter = 0
    #print(imgName)
    #folder_path = os.path.join(my_path, '06192023 SFI renamed')
    for root, dirs, files in os.walk(image_dir):
        #print(root)
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if 'SAGT1' in file_path:
                #print(file_name)
                if imgName.isin([file_name]).any():
                    #print(file_name)
                    shutil.copy(file_path, destination_folder)
                    counter += 1
                    #print("File Path:", file_path)
                    #print(file_name)
    print(f"File Copied ....:{counter}")


def main():

    # get the bounding info into a dataframe
    sagT1_bounding = pd.read_csv(csv_path)

    # get the image names tha has a bounding info
    imgName = sagT1_bounding['image']

    # create_fodler()
    prepareData(imgName)

if __name__ == '__main__':
    my_path = "/Users/jiten/Masters/WorkPlace/"
    folder_path = "/Users/jiten/Masters/WorkPlace/MRI Fractures Project/"

    csv_path = os.path.join(folder_path,'boundingInfo','SAGT1 bounding boxes.csv')
    image_dir = os.path.join(my_path, 'MRI_Image','06192023 SFI renamed')
    destination_folder = os.path.join(folder_path, 'SAGT1_Images')
    #train_dir = os.path.join(folder_path, "train_data")
    #val_dir = os.path.join(folder_path, "validate")
    #test_dir = os.path.join(folder_path, "test_data")
    main()