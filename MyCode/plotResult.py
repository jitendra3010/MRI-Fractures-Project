import os
import matplotlib.pyplot as plt
import pandas as pd

def plotLoss(lossDf):
    """
    Plot the x an y
    """
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss in each epoch")
    plt.plot(lossDf['epoch'], lossDf['loss_val'])
    #lossDf.plot(x='epoch', y='loss_val')
    plt.show()

def main():

    lossDf = pd.read_csv(lossFile)

    plotLoss(lossDf)

if __name__ == '__main__':

    folder_path = os.getcwd() 

    lossFile = os.path.join(folder_path, "LossOutput.csv")
    
    main()