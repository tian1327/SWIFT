import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_loss_acc(path_list):
    # === Global Font Control ===
    FONT_SIZE = 18
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['axes.labelsize'] = FONT_SIZE
    plt.rcParams['xtick.labelsize'] = FONT_SIZE - 2
    plt.rcParams['ytick.labelsize'] = FONT_SIZE - 2
    plt.rcParams['legend.fontsize'] = FONT_SIZE - 1

    # === Load Data ===
    losses, accs = [], []
    for path in path_list:
        df = pd.read_csv(path)
        epoch = df['Epoch'].values
        loss = df['Train_loss'].values
        acc = df['Test_acc'].values
        assert len(epoch) == len(loss) == len(acc)
        losses.append(loss)
        accs.append(acc)

    # === Create Subplots ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # === Define Colors ===
    blue_shades = ['#6baed6', '#3182bd', '#2171b5', '#075fb8', '#08306b']  # light → dark
    # blue_shades = ['white', 'white', 'white', 'white', 'white']  # all white for invisible lines
    
    color_map = {
        1.0: 'tab:red',
        0.1: blue_shades[0],
        0.07: blue_shades[1],
        0.01: blue_shades[3],
        0.001: blue_shades[4],
    }

    LINE_WIDTH = 2.0

    # === Plot Training Loss ===
    # Fixed T_loss (solid)
    ax1.plot(epoch, losses[4], linestyle='-', linewidth=LINE_WIDTH,
             label='No TT ($T_{loss}=1.0$)', color=color_map[1.0])

    ax1.plot(epoch, losses[5], linestyle='-', linewidth=LINE_WIDTH,
             label=r'$T_{loss}=0.1$', color=color_map[0.1])

    ax1.plot(epoch, losses[6], linestyle='-', linewidth=LINE_WIDTH,
             label=r'$T_{loss}=0.07$', color=color_map[0.07])
    
    ax1.plot(epoch, losses[7], linestyle='-', linewidth=LINE_WIDTH,
             label=r'$T_{loss}=0.01$', color=color_map[0.01])
        
    # Learned T_loss (dashed)

    ax1.plot(epoch, losses[0], linestyle='--', linewidth=LINE_WIDTH,
             label=r'$T_{loss}$ init. 1.0', color=color_map[1.0])    

    ax1.plot(epoch, losses[1], linestyle='--', linewidth=LINE_WIDTH,
             label=r'$T_{loss}$ init. 0.1', color=color_map[0.1])    

    ax1.plot(epoch, losses[2], linestyle='--', linewidth=LINE_WIDTH,
             label=r'$T_{loss}$ init. 0.07', color=color_map[0.07])

    ax1.plot(epoch, losses[3], linestyle='--', linewidth=LINE_WIDTH,
             label=r'$T_{loss}$ init. 0.01', color=color_map[0.01])

    ax1.set_xlabel('Training epochs')
    ax1.set_ylabel('Training loss')
    ax1.grid(alpha=0.2)

    # === Plot Test Accuracy ===
    # Learned (dashed)

    ax2.plot(epoch, accs[0], linestyle='--', linewidth=LINE_WIDTH,
             label=r'$T_{loss}$ init. 1.0', color=color_map[1.0])
    ax2.plot(epoch, accs[1], linestyle='--', linewidth=LINE_WIDTH,
             label=r'$T_{loss}$ init. 0.1', color=color_map[0.1])
    ax2.plot(epoch, accs[2], linestyle='--', linewidth=LINE_WIDTH,
             label=r'$T_{loss}$ init. 0.07', color=color_map[0.07])
    ax2.plot(epoch, accs[3], linestyle='--', linewidth=LINE_WIDTH,
             label=r'$T_{loss}$ init. 0.01', color=color_map[0.01])

    # Fixed (solid)
    ax2.plot(epoch, accs[4], linestyle='-', linewidth=LINE_WIDTH,
             label='No TT ($T_{loss}=1.0$)', color=color_map[1.0])
    ax2.plot(epoch, accs[5], linestyle='-', linewidth=LINE_WIDTH,
             label=r'$T_{loss}=0.1$', color=color_map[0.1])
    ax2.plot(epoch, accs[6], linestyle='-', linewidth=LINE_WIDTH,
             label=r'$T_{loss}=0.07$', color=color_map[0.07])
    ax2.plot(epoch, accs[7], linestyle='-', linewidth=LINE_WIDTH,
             label=r'$T_{loss}=0.01$', color=color_map[0.01])

    ax2.axhline(y=53.5, color='black', linestyle='-', alpha=0.5, linewidth=4)
    ax2.text(
        epoch[-1] * 0.2, 52.0, 'Linear Probing Acc.=53.5%',
        fontsize=FONT_SIZE - 3,
        bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', pad=2.0)
    )

    ax2.set_xlabel('Training epochs')
    ax2.set_ylabel('Test acc. (%)')
    ax2.grid(alpha=0.2)

    # === Shared Legend (Top) ===
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=4, bbox_to_anchor=(0.51, 1.15), 
               columnspacing=0.6,
               frameon=True, fontsize=FONT_SIZE - 3)

    # === Final Adjustments ===
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('FSFT_temp_loss_acc_all.png', bbox_inches='tight', dpi=300)
    # plt.savefig('FSFT_temp_loss_acc_all.pdf', bbox_inches='tight', dpi=300)
    print("Saved: FSFT_temp_loss_acc_all.[png/pdf]")

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
