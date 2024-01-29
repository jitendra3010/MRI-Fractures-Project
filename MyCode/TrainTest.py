import os
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

def splitTrainTest():
    """
    Function to split data to train and test folder
    """

    test_size = 0.2

    # Get the list of all files in the original folder
    all_files = os.listdir(source_folder)

    # Split the file list into training and testing sets
    train_files, test_files = train_test_split(all_files, test_size=test_size, 
                                               random_state=3010)
    
    # Move files to the respective folders
    for file in train_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(train_dir, file)
        shutil.copy(source_path, destination_path)

    for file in test_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(test_dir, file)
        shutil.copy(source_path, destination_path)


def main():

    splitTrainTest()

if __name__ == '__main__':
    my_path = "/Users/jiten/Masters/WorkPlace/"
    folder_path = "/Users/jiten/Masters/WorkPlace/MRI Fractures Project/"

    
    source_folder = os.path.join(folder_path, 'SAGT1_Images')
    train_dir = os.path.join(folder_path, "train_data")
    #val_dir = os.path.join(folder_path, "validate")
    test_dir = os.path.join(folder_path, "test_data")
    main()