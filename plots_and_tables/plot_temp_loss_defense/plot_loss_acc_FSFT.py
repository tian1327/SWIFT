import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.variant'] = 'normal'

# === Control Variables ===
FONT_SIZE = 11
LEGEND_SIZE = 10

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # === Plot training loss ===
    # Learned T_loss
    ax1.plot(epoch, losses[0], linestyle='--', label=r'learn $T_{loss}$=1.0', color='tab:red')
    ax1.plot(epoch, losses[1], linestyle='--', label=r'learn $T_{loss}$=0.1', color=blue_shades[0])
    ax1.plot(epoch, losses[2], linestyle='--', label=r'learn $T_{loss}$=0.07', color=blue_shades[1])
    ax1.plot(epoch, losses[3], linestyle='--', label=r'learn $T_{loss}$=0.01', color=blue_shades[2])

    # Fixed T_loss
    # ax1.plot(epoch, losses[4], linestyle='-', label=r'fixed $T_{loss}$=1.0', color='tab:red')
    # ax1.plot(epoch, losses[5], linestyle='-', label=r'fixed $T_{loss}$=0.1', color=blue_shades[0])
    # ax1.plot(epoch, losses[6], linestyle='-', label=r'fixed $T_{loss}$=0.07', color=blue_shades[1])
    # ax1.plot(epoch, losses[7], linestyle='-', label=r'fixed $T_{loss}$=0.01', color=blue_shades[2])

    ax1.set_xlabel('Training epochs', fontsize=FONT_SIZE)
    ax1.set_ylabel('Training loss', fontsize=FONT_SIZE)
    ax1.grid(alpha=0.2)

    # === Plot test accuracy ===
    ax2.plot(epoch, accs[0], linestyle='--', label=r'learn $T_{loss}$=1.0', color='tab:red')
    ax2.plot(epoch, accs[1], linestyle='--', label=r'learn $T_{loss}$=0.1', color=blue_shades[0])
    ax2.plot(epoch, accs[2], linestyle='--', label=r'learn $T_{loss}$=0.07', color=blue_shades[1])
    ax2.plot(epoch, accs[3], linestyle='--', label=r'learn $T_{loss}$=0.01', color=blue_shades[2])

    ax2.plot(epoch, accs[4], linestyle='-', label=r'fixed $T_{loss}$=1.0', color='tab:red')
    ax2.plot(epoch, accs[5], linestyle='-', label=r'fixed $T_{loss}$=0.1', color=blue_shades[0])
    ax2.plot(epoch, accs[6], linestyle='-', label=r'fixed $T_{loss}$=0.07', color=blue_shades[1])
    ax2.plot(epoch, accs[7], linestyle='-', label=r'fixed $T_{loss}$=0.01', color=blue_shades[2])

    ax2.axhline(y=53.5, color='black', linestyle='--')
    ax2.set_xlabel('Training epochs', fontsize=FONT_SIZE)
    ax2.set_ylabel('Test accuracy (%)', fontsize=FONT_SIZE)
    ax2.grid(alpha=0.2)

    # Legend styling
    ax2.legend(
        loc='center right',
        fontsize=LEGEND_SIZE,
        ncol=2,
        framealpha=0.5,
        columnspacing=0.5,
        labelspacing=0.2,
        facecolor='white',
        bbox_to_anchor=(1.01, 0.45)
    )

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
