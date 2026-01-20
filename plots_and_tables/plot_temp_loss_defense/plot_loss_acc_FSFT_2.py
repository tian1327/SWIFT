import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.variant'] = 'normal'

# === Control Variables ===
FONT_SIZE = 12
LEGEND_SIZE = 10
LINE_WIDTH = 2  # control overall line thickness

# Gradually darker blues for T_loss = 0.1, 0.07, 0.01
blue_shades = ['#6baed6', '#3182bd', '#08306b']

def plot_loss_acc(path_list):

    losses = []
    accs = []
    for path in path_list:
        df = pd.read_csv(path)
        epoch = df['Epoch'].values
        loss = df['Train_loss'].values
        acc = df['Test_acc'].values
        assert len(epoch) == len(loss) == len(acc)
        losses.append(loss)
        accs.append(acc)

    # === Two rows, one column ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=False)

    # === Plot training loss (top) ===
    
    # Fixed T_loss (no temperature tuning)
    ax1.plot(epoch, losses[4], linestyle='-', linewidth=LINE_WIDTH,
             label=r'No TT (fixed $T_{loss}$=1.0)', color='tab:red')
    # ax1.plot(epoch, losses[5], linestyle='-', linewidth=LINE_WIDTH,
    #          label=r'fixed $T_{loss}$=0.1', color=blue_shades[0])
    # ax1.plot(epoch, losses[6], linestyle='-', linewidth=LINE_WIDTH,
    #          label=r'fixed $T_{loss}$=0.07', color=blue_shades[1])
    # ax1.plot(epoch, losses[7], linestyle='-', linewidth=LINE_WIDTH,
    #          label=r'fixed $T_{loss}$=0.01', color=blue_shades[2])    
    
    # Learned T_loss
    # ax1.plot(epoch, losses[0], linestyle='-', linewidth=LINE_WIDTH,
    #          label=r'Learnable $T_{loss}$ init. to 1.0', color='tab:red')
    ax1.plot(epoch, losses[1], linestyle='-', linewidth=LINE_WIDTH,
             label=r'Learnable $T_{loss}$ init. to 0.1', color=blue_shades[0])
    ax1.plot(epoch, losses[2], linestyle='-', linewidth=LINE_WIDTH,
             label=r'Learnable $T_{loss}$ init. to 0.07', color=blue_shades[1])
    ax1.plot(epoch, losses[3], linestyle='-', linewidth=LINE_WIDTH,
             label=r'Learnable $T_{loss}$ init. to 0.01', color=blue_shades[2])

    ax1.set_xlabel('Training epochs', fontsize=FONT_SIZE)
    ax1.set_ylabel('Training loss', fontsize=FONT_SIZE)
    # ax1.set_ylim(-0.5, 6)
    ax1.grid(alpha=0.2)
    ax1.legend(
        loc='center right',
        fontsize=LEGEND_SIZE,
        ncol=1,
        framealpha=0.5,
        columnspacing=0.5,
        labelspacing=0.1,
        # facecolor='white',
        bbox_to_anchor=(1.01, 0.40)
    )

    # === Plot test accuracy (bottom) ===
    # ax2.plot(epoch, accs[0], linestyle='-', linewidth=LINE_WIDTH,
    #          label=r'Learnable $T_{loss}$ init. to 1.0', color='tab:red')
    ax2.plot(epoch, accs[1], linestyle='-', linewidth=LINE_WIDTH,
             label=r'Learnable $T_{loss}$ init. to 0.1', color=blue_shades[0])
    ax2.plot(epoch, accs[2], linestyle='-', linewidth=LINE_WIDTH,
             label=r'Learnable $T_{loss}$ init. to 0.07', color=blue_shades[1])
    ax2.plot(epoch, accs[3], linestyle='-', linewidth=LINE_WIDTH,
             label=r'Learnable $T_{loss}$ init. to 0.01', color=blue_shades[2])

    ax2.plot(epoch, accs[4], linestyle='-', linewidth=LINE_WIDTH,
             label=r'fixed $T_{loss}$=1.0', color='tab:red')
    # ax2.plot(epoch, accs[5], linestyle='-', linewidth=LINE_WIDTH,
    #          label=r'fixed $T_{loss}$=0.1', color=blue_shades[0])
    # ax2.plot(epoch, accs[6], linestyle='-', linewidth=LINE_WIDTH,
    #          label=r'fixed $T_{loss}$=0.07', color=blue_shades[1])
    # ax2.plot(epoch, accs[7], linestyle='-', linewidth=LINE_WIDTH,
    #          label=r'fixed $T_{loss}$=0.01', color=blue_shades[2])

    ax2.axhline(y=53.5, color='tab:gray', linestyle='--', alpha=0.7, linewidth=3)
    # add a text box to label the horizontal line
    ax2.text(
        epoch[-1]*0.62, 51.5, 'Linear Probing Acc.=53.5%',
        fontsize=LEGEND_SIZE,
        color='black',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0)
    )

    ax2.set_xlabel('Training epochs', fontsize=FONT_SIZE)
    ax2.set_ylabel('Test acc. (%)', fontsize=FONT_SIZE)
    ax2.grid(alpha=0.2)

    # Legend styling (bottom plot)
    # ax2.legend(
    #     loc='center right',
    #     fontsize=LEGEND_SIZE,
    #     ncol=2,
    #     framealpha=0.5,
    #     columnspacing=0.5,
    #     labelspacing=0.2,
    #     facecolor='white',
    #     bbox_to_anchor=(1.01, 0.45)
    # )

    plt.tight_layout()
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
