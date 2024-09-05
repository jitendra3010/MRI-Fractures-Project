import pandas as pd
import numpy as np
import os
from PIL import Image
import shutil
import random




def prepareData(imgName, img):
    """
    Function to copy the images from main data folder to
    a different folder for all img images
    """

    print("Start of Copy files")
    unhealthy_counter = 0
    healthy_counter = 0
    #print(imgName)
    #folder_path = os.path.join(my_path, '06192023 SFI renamed')
    for root, dirs, files in os.walk(image_dir):
        #print(root)
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if img in file_path:
                #print(file_name)
                # take the unhealty images to differnt folder for training and testing
                if imgName.isin([file_name]).any():
                    #print(file_name)
                    if img == 'SAGT1':
                        shutil.copy(file_path, destination_folder_SAGT1)
                    elif img == 'SAGIR':
                        shutil.copy(file_path, destination_folder_SAGIR)
                    unhealthy_counter += 1
                    #print("File Path:", file_path)
                    #print(file_name)
                
                # take the healthy images to different folder
                if file_name not in imgName.values:
                    # copy only those healthy image slice which has more than 65% of the image as foot
                    h_img = Image.open(file_path)
                    h_img_gray = h_img.convert("L")
                    h_img_array = np.array(h_img_gray)
                    h_size = h_img.size
                    black_pix = np.sum(h_img_array <= 10) # pixel value <=10 considered as black

                    if( (black_pix / (h_size[0] * h_size[1])) < 0.65 ):
                        if img == 'SAGT1':
                            shutil.copy(file_path, healthy_folder_SAGT1)
                        elif img =='SAGIR':
                            shutil.copy(file_path, healthy_folder_SAGIR)
                        healthy_counter +=1

    print(f"healthy File Copied ....:{unhealthy_counter}")
    print(f"healthy files Copied:{healthy_counter}")


def main(img):

    if img == 'SAGT1':
       # get the bounding info into a dataframe
        df_bounding = pd.read_csv(csv_path_SAGT1)
    elif img == 'SAGIR':
        df_bounding = pd.read_csv(csv_path_SAGIR)

    # get the image names tha has a bounding info
    imgName = df_bounding['image']
    
    # create_fodler()
    prepareData(imgName,img)

if __name__ == '__main__':
    my_path = "/Users/jiten/Masters/WorkPlace/"
    folder_path = "/Users/jiten/Masters/WorkPlace/MRI Fractures Project/"

    csv_path_SAGT1 = os.path.join(folder_path,'boundingInfo','SAGT1 bounding boxes.csv')
    csv_path_SAGIR = os.path.join(folder_path,'boundingInfo','SAGIR bounding boxes.csv')
 
    image_dir = os.path.join(my_path, 'MRI_Image','06192023 SFI renamed')
    destination_folder_SAGT1 = os.path.join(folder_path, 'SAGT1_Images')
    destination_folder_SAGIR = os.path.join(folder_path, 'SAGIR_Images')

    healthy_folder_SAGT1 = os.path.join(folder_path, 'healthy_SAGT1')
    healthy_folder_SAGIR = os.path.join(folder_path, 'healthy_SAGIR')

    #train_dir = os.path.join(folder_path, "train_data")
    #val_dir = os.path.join(folder_path, "validate")
    #test_dir = os.path.join(folder_path, "test_data")
    main('SAGIR')