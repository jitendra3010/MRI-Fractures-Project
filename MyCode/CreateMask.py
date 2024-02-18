import pandas as pd
import os
from PIL import Image, ImageDraw
import shutil
import cv2

def main(img):

    # get the bounding info into a dataframe
    if img == 'SAGT1':
       # get the bounding info into a dataframe
        df_bounding = pd.read_csv(csv_path_SAGT1)
    elif img == 'SAGIR':
        df_bounding = pd.read_csv(csv_path_SAGIR)

    train_dir = train_dir_dict[img]
    test_dir = test_dir_dict[img]

    train_mask_dir = train_mask_dir_dict[img]
    test_mask_dir = test_mask_dir_dict[img]

    # create the mask for training data
    createMask(df_bounding, train_dir, train_mask_dir)

    # create the msk for testing data
    createMask(df_bounding, test_dir, test_mask_dir)

def createMask(df_bounding, source, dest):

    counter = 0 
    counter_self = 0
    # as of now we are looking for only Edema labels
    filter = (df_bounding['value_rectanglelabels'] == '[\'Edema\']')
    result_df = df_bounding[filter]

    # Get the list of all files in the folder
    all_files = os.listdir(source)

    #loop through the files and create the masks
    for file in all_files:

        #if file in result_df['image']:
        if (result_df['image'].eq(file)).any():

            condition = result_df['image'] == file
            #print(condition.sum())

             # if there are multiple records then create multiple mask
            for i in range(condition.sum()):
                # get the image dimension
                img_width = result_df[condition].iloc[i]['original_width']
                img_height =  result_df[condition].iloc[i]['original_height']
                #print(len(img_width), file)

           
                img_width = int(img_width)  
                img_height = int(img_height)

                # get the coordinates for the bounding boxes
                x = result_df[condition].iloc[i]['value_x'] * img_width /100
                y = result_df[condition].iloc[i]['value_y'] * img_height /100
                w = result_df[condition].iloc[i]['value_width'] * img_width /100
                h = result_df[condition].iloc[i]['value_height'] * img_height /100
                #x,y,w,h = result_df.loc[file][['','value_y','','value_height']]

                # generate the mask image
                mask_image = generateMask(img_width, img_height, x, y, w, h)

                # change the file name if there are multiple masks
                if(condition.sum() >1):
                    file_name , ext = os.path.splitext(file)

                    file_name = file_name + '_' + str(i)
                    file_name = file_name + ext

                    # save the image
                    mask_image.save(f"{dest}/{file_name}")
                    counter +=1

                    # create copies of the same image when there are multiple mask
                    shutil.copy(f"{source}/{file}",f"{source}/{file_name}")
                    counter_self +=1
                else:
                    mask_image.save(f"{dest}/{file}")
                    counter +=1
    
    print(f"Total Images:::{len(all_files)} ::::: of source :::: {source}")
    print(f"Total Mask created :::{counter} ::::: at destination :::: {dest}")



def generateMask(img_width, img_height, x, y, w, h):
    """
    Function to generate the mask image 
    """

    # Create an empty image with a white background
    mask_image = Image.new("L", (img_width, img_height), 0)  # 'L' mode is for grayscale (0-255)

    # Draw a rectangle on the mask image
    draw = ImageDraw.Draw(mask_image)
    draw.rectangle([x, y, x + w, y + h], fill=255)  # 'fill' sets the color (255 for white)

    return mask_image

if __name__ == '__main__':
    my_path = "/Users/jiten/Masters/WorkPlace/"
    folder_path = os.path.join(my_path,'MRI Fractures Project')

    csv_path_SAGT1 = os.path.join(folder_path,'boundingInfo','SAGT1 bounding boxes.csv')
    csv_path_SAGIR = os.path.join(folder_path,'boundingInfo','SAGIR bounding boxes.csv')
 
    source_folder_dict = {}
    source_folder_SAGT1 = os.path.join(folder_path, 'SAGT1_Images')
    source_folder_SAGIR = os.path.join(folder_path, 'SAGIR_Images')
    source_folder_dict['SAGT1'] = source_folder_SAGT1
    source_folder_dict['SAGIR'] = source_folder_SAGIR

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

    # calling for SAGIR , change this parameter to SAGT1 when required
    main('SAGIR') 



