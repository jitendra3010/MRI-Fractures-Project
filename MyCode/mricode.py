import os
from Agent import Agent

def main(img, train_flag=True):

    if train_flag:
        train_dir = train_dir_dict[img]
        mask_dir = train_mask_dir_dict[img]

        # initialize agent
        agent = Agent(train_flag,img_dir=train_dir,msk_dir=mask_dir,folder_path=folder_path,num_epochs=1, batchSize=30)
        agent.initializeUnet()

        # load custom dataset
        train_loader = agent.loadCustomData()

         # run the model
        loss_df = agent.runModel(train_loader)

         # write the run
        agent.writeRun(loss_df,'Result/LoassOutput.csv')

         # save the net
        agent.save_net(file_name='UNet')

    else:
        test_dir = test_dir_dict[img]
        mask_dir_test = test_mask_dir_dict[img]

        agent = Agent(train_flag,img_dir=test_dir,msk_dir=mask_dir_test,folder_path=folder_path,state='old',num_epochs=1, batchSize=30)
        agent.initializeUnet('UNetMar 29, 2024 06_45PM')
        
        test_loader = agent.loadCustomData()

        predictions = agent.runModel(test_loader)

        print(len(predictions))

        Agent.savePredictions(loader=test_loader, predictions=predictions)
        #test_loader.dataset.image_list

        #mask_dir_test = test_mask_dir_dict[img]

        #test_loader = loadCustomData(test_dir, train_flag=False)

        #runModel(test_loader , train_flag=False)

        # run the model
        

        #print(f"Inside Main::::{train_dir}")



    
if __name__ == '__main__':
    my_path = "/Users/jiten/Masters/WorkPlace/"
    folder_path = os.getcwd() #"/Users/jiten/Masters/WorkPlace/MRI Fractures Project/"
    
    folder_path = os.path.dirname(folder_path)

    print(folder_path)

    #source_folder = os.path.join(folder_path, 'SAGT1_Images')
    train_dir_SAGT1 = os.path.join(folder_path, "train_data_SAGT1")
    test_dir_SAGT1 = os.path.join(folder_path, "test_data_SAGT1")
        
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
    
    print("Call main for SAGIR")
    main('SAGIR',train_flag=False)
