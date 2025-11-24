import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


dataset = ['density', 'pressure', 'CH4', 'temperature',
         'diffusivity', 'QC', 'QGRAUP', 'QG']
dataset_label = ['Density', 'Pressure', 'CH4', 'Temperature',
         'Diffusivity', 'QC', 'QGRAUP', 'QG']
# dataset = ['density']
bits = [64, 64, 64, 32, 64, 32, 32, 32, 32]
csv_files_prefix = "../progressive_residual_"

colors = {
    "cuSZ": "#bc80bd",
    "cuSZp": "#078D07", 
    "cuZFP": "#bb9727",
    "PRISM": "#e41a1c",
    "MDR (cpu)": "#b2b2b2",
    "HP-MDR": "#377eb8",
}
step_compressors = ["cuSZ", "cuSZp", "cuZFP"]
line_compressors = ["PRISM", "MDR (cpu)", "HP-MDR"]

fig, axes = plt.subplots(2, 4, figsize=(32, 12))
axes = axes.flatten()
handles, labels = [], []
for i, data in enumerate(dataset):
    filename = csv_files_prefix + data + ".csv"
    data = pd.read_csv(filename)
    columns_of_interest = ['type', 'error', 'ratio']
    retrieval = data[columns_of_interest]
    ax = axes[i]
    maxbr = 0
    maxeb = 0
    for idx, (comp, group) in enumerate(retrieval.groupby('type')):
        if maxbr < np.max(bits[i]/ group['ratio'][3:9]):
            maxbr = np.max(bits[i]/ group['ratio'][3:9])
        if maxeb < np.max(group['error'][3:9]):
            maxeb = np.max(group['error'][3:9])
        if comp in step_compressors:
            line = ax.step(group['error'][3:9].to_numpy(), bits[i]/ group['ratio'][3:9].to_numpy(), where="post", linewidth=4 ,label=comp, color=colors[comp])
        else:
            line =  ax.plot(group['error'][3:9].to_numpy(), bits[i]/ group['ratio'][3:9].to_numpy(), linewidth=5 , color=colors[comp])
                
        if i == 0:
            if comp in step_compressors:
                handles.append(line[0])
            else:
                handles.append(line[0])
            labels.append(comp)
    ax.invert_xaxis()
    ax.set_xscale("log")
    if i >= 4 :
        ax.set_xlabel("Error Bound", fontsize=26, labelpad=0)
    ax.set_ylabel("Bitrate", fontsize=26, labelpad=12)
    ax.set_title(dataset_label[i], fontsize=26, pad=12) # , fontfamily='Times New Roman'
    if i == 0:
        ax.set_ylim(0, maxbr * 0.22)  # 
    if i == 1:
        ax.set_ylim(0, maxbr * 0.27)
    if i == 2:
        ax.set_ylim(0, maxbr * 0.08)
    if i == 3:
        ax.set_ylim(0, maxbr * 0.4)
    if i == 4:
        ax.set_ylim(0, maxbr * 0.22)
    if i == 5:
        ax.set_ylim(0, maxbr * 0.5)
    if i == 6:
        ax.set_ylim(0, maxbr * 0.3)
    if i == 7:
        ax.set_ylim(0, maxbr * 0.4)
    ax.tick_params(axis='both', which='major', length=8, width=1.2,labelsize=22,pad=12)
    ax.tick_params(axis='both', which='minor', length=4, width=1)
    ax.tick_params(axis='y', which='major',labelsize=22)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
# handles, labels = axes[0, 0].get_legend_handles_labels()
# handles, labels = axes[0, 0].get_legend_handles_labels()
new_order = [3, 4 ,2 ,0 ,1]  
handles = [handles[i] for i in new_order]
labels = [labels[i] for i in new_order]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=5, fontsize=26)

plt.tight_layout()
plt.subplots_adjust(top=0.9,wspace=0.24, hspace=0.45)
plt.savefig('error_br.pdf', dpi=400, bbox_inches='tight')