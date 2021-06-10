
import pandas as pd
import matplotlib.pyplot as plt

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 20}
plt.rc('font', **font)

if __name__ == '__main__':

    # LOSS
    fig=plt.figure()



    #loss = pd.read_csv("~/Desktop/1028/runs/unet/loss.csv")
    loss = pd.read_csv("./runs/unet/loss.csv")

    print(loss.head())
    # plt.plot(loss['epoch'], loss['train_loss'], color='red', label="train_loss", linewidth=2.0)
    # plt.plot(loss['epoch'], loss['test_loss'], color='blue', label="test_loss", linewidth=2.0)
    plt.plot(loss['epoch'], loss['train_acc'], color='red', label="train_acc", linewidth=2.0)
    plt.plot(loss['epoch'], loss['test_acc'], color='blue', label="test_acc", linewidth=2.0)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
    #out_png_path = '.\\loss.pdf'
    #fig.savefig(out_png_path, format='pdf')

    print("ok")
