import os
import shutil
from sklearn.model_selection import train_test_split

def splitTrainTest(img):
    """
    Function to split data to train and test folder
    """

    print("Start of Spit Train/Test")
    counter_trn = 0
    counter_tst = 0
    test_size = 0.2

    # get the source folder name
    source_folder = source_folder_dict[img]

    # get the train and test folder
    train_dir = train_dir_dict[img]
    test_dir = test_dir_dict[img]

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
        counter_trn += 1

    for file in test_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(test_dir, file)
        shutil.copy(source_path, destination_path)
        counter_tst +=1

    print(f"Total train files copied :: {counter_trn}")
    print(f"Total test files copied :: {counter_tst}")


def main(img):

    # split the data for train and test
    splitTrainTest(img)

if __name__ == '__main__':
    my_path = "/Users/jiten/Masters/WorkPlace/"
    folder_path = "/Users/jiten/Masters/WorkPlace/MRI Fractures Project/"

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

    main('SAGIR') # calling for SAGIR , change this parameter to SAGT1 when required