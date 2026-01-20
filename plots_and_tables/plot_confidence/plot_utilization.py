import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

FONT_SIZE = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = FONT_SIZE
plt.rcParams['xtick.labelsize'] = FONT_SIZE - 2
plt.rcParams['ytick.labelsize'] = FONT_SIZE - 2
plt.rcParams['legend.fontsize'] = FONT_SIZE - 1


file_list = [
    'back_up/mask_swift.csv',
    'back_up/loss_OpenCLIP_vit.csv',
    'back_up/loss_INET_vit.csv',
]

# read csv files and extract the Mask column
all_data = []
for file in file_list:
    data = pd.read_csv(file)
    if 'Mask' in data.columns:
        mask_data = data['Mask'].values[:50]
        all_data.append(mask_data)
    else:
        raise ValueError(f"Expected column not found in {file}")

# stack horizontally
combined_data = np.column_stack(all_data)
print(combined_data.shape)
# print the first 5 rows
print(combined_data[:5, :])

# create line plot, x-axis is [1, 51]
plt.figure(figsize=(5, 4))

plt.plot(combined_data[:, 0]*100, label='OpenCLIP w/ Temperature', color='tab:orange', alpha=1.0, linewidth=2)
plt.plot(combined_data[:, 1]*100, label='OpenCLIP', color='tab:blue', alpha=1.0, linewidth=2)
# plt.plot(combined_data[:, 2]*100, label='ImageNet-pretrained', color='tab:green', alpha=0.8, linewidth=2)

# set x ticks to be 1, 2, ..., 50
# plt.xlim(1, 50)
plt.xticks(np.arange(0, 51, 20))
plt.ylim(-5, 100)

# set y ticks every 25
plt.yticks(np.arange(0, 101, 25))

# set x label
plt.xlabel('Training epochs')

plt.ylabel('Utilization (%)')

# add legend
plt.legend(loc='center left', fontsize=14)
plt.grid(alpha=0.2)

plt.tight_layout()

# save plot
plt.savefig("utilization_rate.png", dpi=300)
