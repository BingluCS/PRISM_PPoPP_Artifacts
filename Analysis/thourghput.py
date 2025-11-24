import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_files_prefix = "../progressive_residual_"
datasets = ['density', 'pressure', 'CH4', 'temperature']
datasets_label = ['Density', 'Pressure', 'CH4', 'Temperature']
# datasets = ['density']
colors = ["#fceac5",'#118ab2', '#06d6a0', '#0a495e', '#ef476f', '#8c564b']
fig, axes = plt.subplots(2, 4, figsize=(18, 6))
compressor = ['cuSZp', 'cuSZ', 'cuZFP', 'HP-MDR', 'PRISM']
bar_width = 0.17
colors = {
    "cuSZp": "#fceac5", 
    "cuZFP": "#118ab2",
    "cuSZ": "#06d6a0",
    "HP-MDR": "#0a495e",
    "PRISM": "#ef476f",
}
location = {
    "cuSZp": 0, 
    "cuZFP": 1,
    "cuSZ": 2,
    "HP-MDR": 3,
    "PRISM": 4,
}
error_bounds_d = ['1e-5', '1e-6', '1e-7','1e-8']
error_bounds_f = ['1e-3', '5e-4', '1e-4','1e-5']

for i, data in enumerate(datasets):
    ax_comp = axes[0, i] 
    ax_decomp = axes[1, i]
    filename = csv_files_prefix + data + ".csv"
    columns_of_interest = ['type', 'error', 'comp_th', 'decomp_th']
    data = pd.read_csv(filename)
    retrieval = data[columns_of_interest]
    for idx, (comp, group) in enumerate(retrieval.groupby('type')):
        comp_throughput = group['comp_th'][4:8]
        x_positions = np.arange(len(group['error'][4:8]))
        x_pos = x_positions + location[comp] * bar_width - (len(compressor) - 1) * bar_width / 2
        ax_comp.bar(x_pos, comp_throughput, bar_width, color=colors[comp], alpha=0.8,edgecolor='black', linewidth=1.5,label=comp if i == 0 else "")
    
    for idx, (comp, group) in enumerate(retrieval.groupby('type')):
        comp_throughput = group['decomp_th'][4:8]
        x_positions = np.arange(len(group['error'][4:8]))
        x_pos = x_positions + location[comp] * bar_width - (len(compressor) - 1) * bar_width / 2
        ax_decomp.bar(x_pos, comp_throughput, bar_width, color=colors[comp], alpha=0.8,edgecolor='black', linewidth=1.5)
    
    ax_comp.set_title(datasets_label[i], fontsize=22, pad=8)
    ax_comp.set_ylim(0, 150)
    ax_decomp.set_ylim(0, 150)
    
    for spine in ax_decomp.spines.values():
        spine.set_linewidth(1.5)  
    for spine in ax_comp.spines.values():
        spine.set_linewidth(1.5) 
    
    if i == 0:
        ax_comp.set_ylabel('Comp (GB/s)', fontsize=18,labelpad=8)
        ax_decomp.set_ylabel('Decomp (GB/s)', fontsize=18, labelpad=8)
    else:
        ax_comp.set_yticklabels([])
        ax_decomp.set_yticklabels([])
    
    if i < 3:
        ax_comp.set_xticks(x_positions)
        ax_comp.set_xticklabels(error_bounds_d, fontsize=10)
        ax_decomp.set_xticks(x_positions)
        ax_decomp.set_xticklabels(error_bounds_d, fontsize=10)
    else:
        ax_comp.set_xticks(x_positions)
        ax_comp.set_xticklabels(error_bounds_f, fontsize=10)
        ax_decomp.set_xticks(x_positions)
        ax_decomp.set_xticklabels(error_bounds_f, fontsize=10)
    
    ax_comp.grid(True, alpha=0.3, axis='y')
    ax_decomp.grid(True, alpha=0.3, axis='y')
    
    ax_comp.set_yticks([0, 50, 100, 150])
    ax_decomp.set_yticks([0, 50, 100, 150])
    
    ax_comp.tick_params(axis='both', which='major', length=4, width=1,labelsize=16,pad=5)
    ax_decomp.tick_params(axis='both', which='major', length=4, width=1,labelsize=16,pad=5)

handles, labels = axes[0, 0].get_legend_handles_labels()
handles, labels = axes[0, 0].get_legend_handles_labels()
new_order = [3, 4 ,2 ,0 ,1]  
handles = [handles[i] for i in new_order]
labels = [labels[i] for i in new_order]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
           ncol=len(compressor), fontsize=16, frameon=True, fancybox=True)

fig.suptitle('(a) NVIDIA H100', fontsize=18, y=0.07)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.85, hspace=0.25, wspace=0.08)
plt.savefig('h100_throughput.pdf', dpi=400)