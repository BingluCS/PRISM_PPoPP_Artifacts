import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv_files_prefix = "../progressive_residual_"
datasets = ['density', 'pressure', 'CH4', 'temperature']
bits = [64, 64, 64, 32, 64, 32, 32, 32, 32]
datasets_label = ['Density', 'Pressure', 'CH4', 'Temperature']
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
    
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.45)
colors = {
    "cuSZp": "#93999f", 
    "cuZFP": "#3C4248",
    "cuSZ": "#076da3",
    "HP-MDR": "#00a6fb",
    "PRISM": "#ef233c",
}
location = {
    "cuSZp": 0, 
    "cuZFP": 1,
    "cuSZ": 2,
    "HP-MDR": 3,
    "PRISM": 4,
}
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
# colors = ['#93999f', '#3C4248', '#076da3', '#00a6fb', '#ef233c']
markers = ['o', 's', '^', 'v', 'D']
compressor = ['cuSZp', 'cuZFP', 'cuSZ', 'HP-MDR', 'PRISM']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]


for i, data in enumerate(datasets):
    ax = axes[i // 2, i % 2]
    filename = csv_files_prefix + data + ".csv"
    columns_of_interest = ['type', 'error', 'ratio', 'psnr']
    data = pd.read_csv(filename)
    retrieval = data[columns_of_interest]
    for idx, (comp, group) in enumerate(retrieval.groupby('type')):
        br = bits[i]/ group['ratio']
        psnr = group['psnr']
        ax.plot(br.to_numpy(), psnr.to_numpy(), color=colors[comp], marker=markers[i], linewidth=2, markersize=4, linestyle='--', label=comp if i == 0 else "") 
    ax.set_xlabel('Bitrate', fontsize=18,labelpad=8)  
    ax.set_ylabel('PSNR (dB)', fontsize=18, labelpad=8)    
    ax.set_title(datasets_label[i], fontsize=20, pad=10)
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
    if i == 0:
        ax.set_xlim(left=0, right=12)
        ax.set_ylim(0, 200)
    if i == 1:
        ax.set_xlim(left=0, right=12)
        ax.set_ylim(0, 200)
    if i == 2:
        ax.set_xlim(left=0, right=10)
        ax.set_ylim(0, 200)
    if i == 3:
        ax.set_xlim(left=0, right=10)
        ax.set_ylim(0, 120)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
handles, labels = axes[0, 0].get_legend_handles_labels()
new_order = [3, 4 ,2 ,0 ,1]  
handles = [handles[i] for i in new_order]
labels = [labels[i] for i in new_order]
fig.legend(handles, labels, 
          loc='upper center', 
          bbox_to_anchor=(0.5, 1),
          ncol=5, 
          fontsize=14,
          frameon=True,
          fancybox=True,
          shadow=False,
          columnspacing=1.5)
fig.savefig('rate_dist.pdf', dpi=400, bbox_inches='tight')
# print("图像已保存为 /root/ppopp/plot/rate_dist.pdf")