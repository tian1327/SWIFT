import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_loss_acc(path_list):

    losses = []
    accs = []
    for path in path_list:
        # read the csv file
        df = pd.read_csv(path)
        # extract the loss and accuracy columns
        epoch = df['Epoch'].values
        loss = df['Train_loss'].values
        acc = df['Test_acc'].values

        # assert the length of epoch, loss, acc are the same
        assert len(epoch) == len(loss) == len(acc)

        # append to the list
        losses.append(loss)
        accs.append(acc)

     # plot the losses in the left subfig, accuracies in the right subfig
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.plot(epoch, losses[0], label=f'learn temp, init temp = 0.07', color='green')
    ax1.plot(epoch, losses[1], label=f'no learn temp, init temp = 1.0', color='red')
    ax1.plot(epoch, losses[2], label=f'no learn temp, init temp = 0.01', color='blue')
    # ax1.set_title('OpenCLIP')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.legend()

    ax2.plot(epoch, accs[0], label=f'learn temp, init temp = 0.07', color='green')
    ax2.plot(epoch, accs[1], label=f'no learn temp, init temp = 1.0', color='red')
    ax2.plot(epoch, accs[2], label=f'no learn temp, init temp = 0.01', color='blue')
    # plot a horizontal dashed black line at y=53.4, labeled "Linear probing"
    ax2.axhline(y=53.4, color='black', linestyle='--', label='Linear Probing (Accuracy = 53.4)')

    # ax2.set_title('OpenCLIP')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('learntemp_loss_acc.png')



if __name__ == "__main__":

    path_list = [
        # openclip learn temp, initial temp = 0.07
        "/scratch/user/ltmask/SSL/output/FSFT_openclip_learntemp_inittemp0.07_vitb32_openclip_laion400m_50epochs/output_semi-aves/FSFT_openclip_learntemp_inittemp0.07_semi-aves_finetune_fewshot_REAL-Prompt_16shots_seed1/loss.csv",
        # openclip no learn temp, initial temp = 1.0
        "/scratch/user/ltmask/SSL/output/FSFT_openclip_nolearntemp_inittemp1.0_vitb32_openclip_laion400m_50epochs/output_semi-aves/FSFT_openclip_nolearntemp_inittemp1.0_semi-aves_finetune_fewshot_REAL-Prompt_16shots_seed1/loss.csv",
        # openclip no learn temp, initial temp = 0.01
        "/scratch/user/ltmask/SSL/output/FSFT_openclip_nolearntemp_vitb32_openclip_laion400m_50epochs/output_semi-aves/FSFT_openclip_nolearntemp_semi-aves_finetune_fewshot_REAL-Prompt_16shots_seed1/loss.csv",
    ]

    plot_loss_acc(path_list)
