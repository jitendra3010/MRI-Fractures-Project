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
    counter_val = 0
    test_size = 0.2
    val_size = 0.2

    # get the source folder name
    source_folder = source_folder_dict[img]

    # get the train and test folder
    train_dir = train_dir_dict[img]
    test_dir = test_dir_dict[img]
    val_dir = val_dir_dict[img]

    # Generate a list of strings from P001 to P096
    patient_list = [f'P{str(i).zfill(3)}' for i in range(1, 97)]

    # Split the data into training, testing, and validation sets
    train_ptn, test_ptn = train_test_split(patient_list, test_size=0.2, random_state=3010)
    train_ptn, val_ptn = train_test_split(train_ptn, test_size=0.2, random_state=3010)




    # Get the list of all files in the original folder
    all_files = os.listdir(source_folder)

    # # Split the file list into training and testing sets
    # train_files, test_files = train_test_split(all_files, test_size=test_size, 
    #                                            random_state=3010)
    
    # # split the file list into training and validation
    # train_files, val_files = train_test_split(train_files, test_size=val_size, random_state=3010)
    
    # Move files to the respective folders
    for file in all_files:
        if file[:4] in train_ptn:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(train_dir, file)
            shutil.copy(source_path, destination_path)
            counter_trn += 1
        elif file[:4] in test_ptn:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(test_dir, file)
            shutil.copy(source_path, destination_path)
            counter_tst +=1
        elif file[:4] in val_ptn:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(val_dir, file)
            shutil.copy(source_path, destination_path)
            counter_val +=1

    # for file in all_files:
    #     if file[:4] in test_ptn:
    #         source_path = os.path.join(source_folder, file)
    #         destination_path = os.path.join(test_dir, file)
    #         shutil.copy(source_path, destination_path)
    #         counter_tst +=1
    
    # for file in all_files:
    #     if file[:4] in val_ptn:
    #         source_path = os.path.join(source_folder, file)
    #         destination_path = os.path.join(val_dir, file)
    #         shutil.copy(source_path, destination_path)
    #         counter_val +=1

    print(f"Total train files copied :: {counter_trn}")
    print(f"Total test files copied :: {counter_tst}")
    print(f"Total validaiton files copied :: {counter_val}")


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
    test_dir_dict = {}
    val_dir_dict = {}

    #val_dir = os.path.join(folder_path, "validate")
    train_dir_SAGT1 = os.path.join(folder_path, "train_data_SAGT1")
    test_dir_SAGT1 = os.path.join(folder_path, "test_data_SAGT1")
    val_dir_SAGT1 = os.path.join(folder_path, "val_data_SAGT1")

    train_dir_SAGIR = os.path.join(folder_path, "train_data_SAGIR")
    test_dir_SAGIR = os.path.join(folder_path, "test_data_SAGIR")
    val_dir_SAGIR = os.path.join(folder_path, "val_data_SAGIR")


    train_dir_dict['SAGT1'] = train_dir_SAGT1
    train_dir_dict['SAGIR'] = train_dir_SAGIR

    test_dir_dict['SAGT1'] = test_dir_SAGT1
    test_dir_dict['SAGIR'] = test_dir_SAGIR

    val_dir_dict['SAGT1'] = val_dir_SAGT1
    val_dir_dict['SAGIR'] = val_dir_SAGIR

    main('SAGIR') # calling for SAGIR , change this parameter to SAGT1 when required