import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

datasets = ['density', 'pressure', 'CH4', 'temperature',
         'diffusivity', 'QC', 'QGRAUP', 'QG']
bits = [64, 64, 64, 32, 64, 32, 32, 32, 32]
datasets_label = ['Density', 'Pressure', 'CH4', 'Temperature']

csv_files_prefix = "../ratio_"


for i, filed in enumerate(datasets):
    filename = csv_files_prefix + filed + ".csv"
    columns_of_interest = ['type', 'error', 'ratio']
    data = pd.read_csv(filename)
    retrieval = data[columns_of_interest].pivot(index='error', columns='type', values='ratio')
    print("-----------------------------")
    print(filed)
    print(retrieval)