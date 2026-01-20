import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.variant'] = 'normal'

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # plot the training loss in the left subfig
    ax1.plot(epoch, losses[0], linestyle='--', label=r'learn $T_{s}$=1.0', color='tab:red')
    ax1.plot(epoch, losses[1], linestyle='--', label=r'learn $T_{s}$=0.1', color='tab:green')
    ax1.plot(epoch, losses[2], linestyle='--', label=r'learn $T_{s}$=0.07', color='tab:blue')
    ax1.plot(epoch, losses[3], linestyle='--', label=r'learn $T_{s}$=0.01', color='tab:orange')

    ax1.plot(epoch, losses[4], linestyle='-', label=r'fixed $T_{s}$=1.0', color='tab:red')
    ax1.plot(epoch, losses[5], linestyle='-', label=r'fixed $T_{s}$=0.1', color='tab:green')
    ax1.plot(epoch, losses[6], linestyle='-', label=r'fixed $T_{s}$=0.07', color='tab:blue')
    ax1.plot(epoch, losses[7], linestyle='-', label=r'fixed $T_{s}$=0.01', color='tab:orange')

    # ax1.set_title('OpenCLIP')
    ax1.set_xlabel('Training epochs', fontsize=11)
    ax1.set_ylabel('Training loss', fontsize=11)
    ax1.grid(alpha=0.2)
    # ax1.legend()
    # ax1.legend(loc='center right', fontsize=8, ncol=2)


    # test acc
    ax2.plot(epoch, accs[0], linestyle='--', label=r'learn $T_{s}$=1.0', color='tab:red')
    ax2.plot(epoch, accs[1], linestyle='--', label=r'learn $T_{s}$=0.1', color='tab:green')
    ax2.plot(epoch, accs[2], linestyle='--', label=r'learn $T_{s}$=0.07', color='tab:blue')
    ax2.plot(epoch, accs[3], linestyle='--', label=r'learn $T_{s}$=0.01', color='tab:orange')

    ax2.plot(epoch, accs[4], linestyle='-', label=r'fixed $T_{s}$=1.0', color='tab:red')
    ax2.plot(epoch, accs[5], linestyle='-', label=r'fixed $T_{s}$=0.1', color='tab:green')
    ax2.plot(epoch, accs[6], linestyle='-', label=r'fixed $T_{s}$=0.07', color='tab:blue')
    ax2.plot(epoch, accs[7], linestyle='-', label=r'fixed $T_{s}$=0.01', color='tab:orange')

    # ax2.axhline(y=53.5, color='black', linestyle='--', label='linear probing (Acc = 53.5)')
    ax2.axhline(y=53.5, color='black', linestyle='--')


    # ax2.set_title('OpenCLIP')
    ax2.set_xlabel('Training epochs', fontsize=11)
    ax2.set_ylabel('Test accuracy (%)', fontsize=11)
    ax2.grid(alpha=0.2)
    # make the legend two columns, make the background white
    ax2.legend(loc='center right', fontsize=10, ncol=2, framealpha=0.5,
               columnspacing=0.5, labelspacing=0.2, facecolor='white',
               bbox_to_anchor=(1.01, 0.45),)
    # ax2.legend()

    plt.tight_layout()
    # plt.savefig('FSFT_temp_loss_acc.png')
    plt.savefig('FSFT_temp_loss_acc.pdf', dpi=300)




if __name__ == "__main__":

    path_list = [
        'data/FSFT_learn_1.0.csv',
        'data/FSFT_learn_0.1.csv',
        'data/FSFT_learn_0.07.csv',
        'data/FSFT_learn_0.01.csv',
        'data/FSFT_nolearn_1.0.csv',
        'data/FSFT_nolearn_0.1.csv',
        'data/FSFT_nolearn_0.07.csv',
        'data/FSFT_nolearn_0.01.csv',
    ]

    plot_loss_acc(path_list)
