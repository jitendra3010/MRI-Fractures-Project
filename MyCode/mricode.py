import os
from Agent import Agent
from itertools import chain
from PIL import Image
import numpy as np
from plotResult import plotOptimalThresh, plotTest_IOU

def main(img, train_flag=True):

    if train_flag:
        train_dir = train_dir_dict[img]
        mask_dir = train_mask_dir_dict[img]

        # initialize agent
        agent = Agent(train_flag,img_dir=train_dir,msk_dir=mask_dir,folder_path=folder_path,num_epochs=200, batchSize=30, bilinear=False)
        agent.initializeUnet()

        # load custom dataset
        train_loader = agent.loadCustomData()

         # run the model
        loss_df, prediction_batch, iou_sore_df = agent.runModel(train_loader)

        # join the list of lists
        predictions = list(chain(*prediction_batch))

        # plot few results
        # agent.printPrediction(loader=train_loader, preds=predictions)

        # # write the iou_Score
        w_path = os.path.join(folder_path,"Result","IoUScore.csv")
        agent.writeRun(iou_sore_df,w_path)

        # # write the loss data
        w_path = os.path.join(folder_path,"Result","LossOutput.csv")
        agent.writeRun(loss_df,w_path)

        # # save the net
        agent.save_net(file_name='UNet')
        
        # compute an optimal threshold
        #iou_vs_thresh = agent.optimalThresVsIoU(train_loader, predictions)
        #plotOptimalThresh(iou_vs_thresh)
    else:
        test_dir = test_dir_dict[img]
        mask_dir_test = test_mask_dir_dict[img]

        agent = Agent(train_flag,img_dir=test_dir,msk_dir=mask_dir_test,folder_path=folder_path,state='old',num_epochs=1, batchSize=240, bilinear=False)
        agent.initializeUnet('UNetMay 14, 2024 04_14PM')
        
        test_loader = agent.loadCustomData()

        prediction_batch, iou_score_batch, iou_score_each = agent.runModel(test_loader)

         # join the list of lists
        predictions = list(chain(*prediction_batch))

        # plot few results
        #agent.printPrediction(loader=test_loader, preds=predictions)
        #print(f"The Iou Score for Testing ::{avg_iou_batch}")
        print(len(iou_score_each))
        plotTest_IOU(iou_score_each)
         # # write the iou_Score
        #w_path = os.path.join(folder_path,"Result","IoUScore_test.csv")
        #agent.writeRun(iou_score_batch,w_path)

        #agent.savePredictions(loader=test_loader, predictions=predictions)

    
if __name__ == '__main__':
    my_path = "/Users/jiten/Masters/WorkPlace/"
    folder_path = os.getcwd() #"/Users/jiten/Masters/WorkPlace/MRI Fractures Project/"
    
    ##folder_path = os.path.dirname(folder_path)

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