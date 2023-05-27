from os import path

import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DIR = path.dirname(path.realpath(__file__))
df = pd.read_csv(path.join(CURRENT_DIR, 'input', 'auc_scores.csv'))
df.set_index(df.columns[0], inplace=True)
models = df.index.tolist()
columns = ['Incidence (leaves)', 'Incidence (bunches)', 'Severity (leaves)', 'Severity (bunches)']

fig, axs = plt.subplots(1, len(columns), figsize=(16, 3), sharex=True)
for i, column in enumerate(columns):
    values = df[column].tolist()
    for j, model in enumerate(df.index):
        axs[i].plot([0, values[j]], [j, j], linewidth=3, color='blue')
        axs[i].scatter(values[j], j, color='blue', s=80, zorder=10)

    # Set the y-axis tick labels and limit
    axs[i].set_yticks(range(len(df.index)))
    axs[i].set_yticklabels(df.index)
    axs[i].set_ylim([-0.5, len(df.index) - 0.5])
    axs[i].set_xlim([0, 1.0])
    # Set the title for the subplot
    axs[i].set_title(column, fontsize=12)

plt.suptitle('Model AUC Scores\nClimate Inputs (including onset date)', fontsize=16)
plt.tight_layout()
plt.savefig(path.join(CURRENT_DIR, 'output', 'auc_scores.png'))