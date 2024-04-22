import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Plot customizations
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12.0
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['savefig.bbox'] = 'tight'

layers = np.arange(1, 11)

results_cell = pd.read_pickle('cell_results.pkl')
results_mesh = pd.read_pickle('mesh_results.pkl')


fig, axs = plt.subplots(2, 3, figsize=(22, 10), sharex=True)
axs = axs.ravel()
for i in range(6):
    axs[i].semilogy(layers, results_cell[f'det_{i}'], marker='o', label='Cell-based')
    axs[i].semilogy(layers, results_mesh[f'det_{i}'], marker='x', label='Mesh-based')
    if i > 2:
        axs[i].set_xlabel('Number of layers')
    if i % 3 == 0:
        axs[i].set_ylabel('Decay photon flux [cm/sec]')
    axs[i].grid(True, which='both')
    axs[i].legend()
plt.show()
#fig.savefig('cell_mesh_r2s_comparison.pdf')
