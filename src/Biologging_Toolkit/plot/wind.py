import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
plt.rcParams.update({
    "text.usetex": True,                # Enable LaTeX text rendering
    "font.family": "serif",             # Use a serif font
    "font.serif": ["Computer Modern"],  # Set font to Computer Modern (LaTeX default)
})
from Biologging_Toolkit.applications.Wind import Wind

def plot_contour_upwards_downwards(depids, save = False, save_path = '.') :
    fig, ax = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(6, 6))
    ax = ax.flatten()
    for i, depid in enumerate(depids):
        df = pd.read_csv(f'D:/individus_brut/individus/{depid}/{depid}_dive.csv')
        _df = df[['upwards_mean_5000', 'downwards_mean_5000', 'wind_speed']]
        _df = _df.melt(id_vars='wind_speed').dropna()
        _df.columns = ['Wind speed (m/s)', 'variable', r'PSD at 5kHz (dB re 1 $\mu$Pa)']
        sns.kdeplot(
            data=_df,
            x='Wind speed (m/s)',
            y=r'PSD at 5kHz (dB re 1 $\mu$Pa)',
            hue="variable",
            ax=ax[i],
            legend = False)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
    fig.text(0.56, 0.04, 'Wind speed (m/s)', ha='center', va='center', fontsize=14)
    fig.text(0.04, 0.5, r'PSD at 5kHz (dB re 1 $\mu$Pa)', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.tight_layout(rect=[0.05, 0.05, 1, 1])
    if save :
        fig.savefig(save_path)


def plot_accuracy_frequencies(depids, freqs, path = None, save = False, save_path = '.') :
    if path :
        data = np.load('data_frequency.npy')
    else :
        data = np.full((12,13,2), np.nan)
        for j, freq in enumerate(freqs):
            for i, depid in enumerate(depids) :
                inst = Wind(depids, path = [os.path.join(path, depid) for depid in depids], data = f'downwards_mean_{int(freq)}', test_depid = depid)
                inst.depid_fit()
                data[i,j,0] = inst.wind_model_stats['depid_mae']
                inst = Wind(depids, path = [os.path.join(path, depid) for depid in depids], data = f'downwards_mean_{int(freq)}', test_depid = depid, method = 'Hildebrand')
                inst.depid_fit()
                data[i,j,1] = inst.wind_model_stats['depid_mae']
    fig, ax = plt.subplots(1,2, sharex = True, sharey = True, figsize = (6,5))
    colors = ['#1f77b4', '#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2',  '#7f7f7f','#bcbd22','#17becf','#ffbb78','#98df8a']
    for i in range(12):
        ax[0].plot(freqs, data[i,:,0], '--o', color = colors[i], label = depids[i], alpha = 0.4)
        ax[1].plot(freqs, data[i,:,1], '--o', color = colors[i], alpha = 0.4)
    ax[0].plot(freqs, np.mean(data[:,:,0], axis = 0), linewidth = 3, color = 'crimson')
    ax[1].plot(freqs, np.mean(data[:,:,1], axis = 0), linewidth = 3, color = 'crimson', label = 'average mae')
    ax[0].legend(loc='upper center', ncol=2, fontsize='small')
    ax[0].grid()
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Mean Absolute Error (m/s)')
    fig.tight_layout()
    if save :
        fig.savefig(save_path)
