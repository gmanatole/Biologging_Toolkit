from Biologging_Toolkit.utils.plot_utils import subplots_centered
from Biologging_Toolkit.utils.mixed_layer_depth_utils import get_previous_wind, get_wind_gust
from Biologging_Toolkit.utils.format_utils import numpy_fill
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
import numpy as np
import os
import seaborn as sns
from scipy.signal import find_peaks

plt.rcParams.update({
    "text.usetex": True,                # Enable LaTeX text rendering
    "font.family": "serif",             # Use a serif font
    "font.serif": ["Computer Modern"],  # Set font to Computer Modern (LaTeX default)
})
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def plot_wind_correlation(depids, path, data : list = ['lstm'], labels = None, save = False, save_path = '.') :
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = colormaps.get_cmap('viridis').resampled(len(data))
    markers = ['s', 'P', '^', 'o']
    ax.set_title(f"Correlation for all FODs")
    ax.grid()
    labels = data if labels is None else labels
    df = pd.DataFrame()
    for i, depid in enumerate(depids):
        _df = pd.read_csv(os.path.join(path, depid, f'{depid}_dive.csv'))
        _df = get_previous_wind(_df, data=data)
        df = pd.concat((df, _df))
    legend_handles = []
    for j, (_data, label) in enumerate(zip(data, labels)):
        cols = ['meop_mld']
        for i in range(48):
            cols.append(f'{_data}_{i}h')
        corr_df = df[cols]
        ax.plot(list(range(0, 48)), corr_df.corr().iloc[1:, 0], color=colors(j), marker=markers[j])
        legend_handles.append(plt.Line2D([0], [0], color=colors(j), marker=markers[j], lw = 2, label = label))
    ax.legend(handles=legend_handles)
    fig.text(0.56, 0.04, 'Hours before MLD obtention', ha='center', va='center', fontsize=14)
    fig.text(0.04, 0.5, r'Pearson correlation coefficient with previous wind speeds and MLD', ha='center', va='center',
             rotation='vertical', fontsize=14)
    if save :
        fig.savefig(os.path.join(save_path, f'global_wind_correlation.png'))

def plot_wind_average_correlation(depids, path, data = 'lstm', group = None, save = False, save_path = '.') :
    #### HEATMAP FOR BINNED INDEPENDANT VARIABLE OF CORRELATION AS A FUNCTION OF WIND AVERAGING PERIOD AND HOURS BEFORE MLD
    df = pd.DataFrame()
    for i, depid in enumerate(depids):
        _df = pd.read_csv(os.path.join(path, depid, f'{depid}_dive.csv'))
        _df = get_previous_wind(_df, data = data)
        cols = ['meop_mld', group] if group else ['meop_mld']
        for i in range(48):
            cols.append(f'{data}_{i}h')
        _df = _df[cols]
        for avg in range(2, 25):
            _roll = _df.iloc[:, len(cols)-48:len(cols)].rolling(avg, min_periods=1, center=True).mean()
            _roll.columns = [f'avg{avg}_{col}' for col in list(_roll.columns)]
            _df = pd.concat((_df, _roll), axis=1)
        df = pd.concat((df, _df))
    if group:
        bins = [-np.inf, df[group].quantile(0.33), df[group].quantile(0.66), np.inf]
        labels = list(range(len(bins) - 1))
        df['bins'] = pd.cut(df[group], bins=bins, labels=labels, right=False)
        fig, ax = subplots_centered(2, 2, nfigs = 3, figsize=(12, 12), sharey=True)
        for k in range(len(bins)-1):
            sns.heatmap(
                df[df.bins == k].corr().iloc[2:-1, 0].to_numpy().reshape(-1, 48),
                ax=ax[k],
                cbar_kws={'label': 'Pearson correlation coefficient'}
            )
            ax[k].set_title(f'Bin {k}')
    else:
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(
            df.corr().iloc[1:,0].to_numpy().reshape(-1, 48),
            ax=ax,
            cbar_kws={'label': 'Pearson correlation coefficient'}
        )
    fig.text(0.5, 0.04, 'Hours before MLD obtention', ha='center', va='center', fontsize=14)
    fig.text(0.04, 0.5, 'Wind averaging period in hours', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.tight_layout(rect=[0.05, 0.05, 1, 1])
    if save :
        if group :
            fig.savefig(os.path.join(save_path, f'correlation_{data}_mld_bins_{group}.pdf'))
        else :
            fig.savefig(os.path.join(save_path, f'correlation_{data}_mld.pdf'))

def scatter_wind_mld(df, vars, save = False, save_path = '.') :
    len_vars = len(vars.keys())
    if len_vars % 2 == 0 :
        fig, ax = plt.subplots(len_vars // 2, 2, figsize=(10, 10))
        ax = ax.flatten()
    else :
        fig, ax = subplots_centered(len_vars // 2 + 1, 2, nfigs = len_vars, figsize=(10, 10))
    for i, key in enumerate(vars.keys()):
        scat = ax[i].scatter(df.peaks.to_numpy(), df.mld.to_numpy(), c=df[key].to_numpy())
        fig.colorbar(scat, ax=ax[i])
        ax[i].set_title(vars[key])
    for a in ax:
        a.set_xlabel("")
        a.set_ylabel("")
    fig.text(0.5, 0.0, "Wind Speed (m/s)", ha="center", fontsize=12)
    fig.text(0.0, 0.5, "MLD (m)", va="center", rotation="vertical", fontsize=12)
    fig.tight_layout()
    if save :
        fig.savefig(os.path.join(save_path, f'wind_speed_mld_visualisation.pdf'))

def plot_regression_results(inst, labels, model = 'OLS', save = False, save_path = '.', **kwargs) :
    ### PLOT COEFFICIENTS, R2, PVALUES AND MODEL PREDICTION AS A FUNCTION OF TIME LAG
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    colors = ['blue', 'red', 'fuchsia', 'gold', 'green','cyan','magenta','lime']
    markers = ['o','p','s','x','^','v','+']
    lines = []
    ax1 = ax[0].twinx()
    results = getattr(inst, f'{model}_results')
    for i, val in enumerate(results.keys()) :
        if val == 'const':
            continue
        if val == 'gradient' :
            line = ax1.plot(results[val][0], colors[i], marker = markers[i], label = labels[val], markevery = 2)
            ax1.set_ylabel('Gradient coefficient', color=colors[i])
            ax1.tick_params(axis='y', labelcolor=colors[i])
            lines.extend(line)
            continue
        line = ax[0].plot(results[val][0], colors[i], marker = markers[i], label = labels[val], markevery = 2)
        lines.extend(line)
    ax0_labels = [l.get_label() for l in lines]
    ax[0].legend(lines, ax0_labels, loc='upper left')
    ax[0].grid()
    ax[0].set_xlabel(f'Time differential between wind gust and {inst.target.upper()} (h)')
    ax[0].set_title('Linear regression coefficients')

    r_squared = getattr(inst, f'{model}_r_squared')
    ax[1].plot(r_squared, color='green', label=r'Global $R^2$', marker='^')
    ax[1].plot([inst.time_diff]*2, [0, r_squared[inst.time_diff]], '--', c = 'k')
    ax[1].set_ylabel(r'$R^2$')
    ax[1].set_xlabel('Time differential between wind gust and MLD (h)')
    ax[1].legend()
    ax[1].set_title(r'$R^2$ between dependant variables and $\Delta$MLD')
    ax[1].grid()

    lines = []
    for i, val in enumerate(results.keys()) :
        if val == 'const':
            continue
        line = ax[2].plot(results[val][1], colors[i], marker = markers[i], label = labels[val], markevery = 2)
        lines.extend(line)
    ax[2].set_xlabel('Time differential between wind gust and MLD (h)')
    ax[2].set_title('P-values for wind gust speed and duration')
    ax[2].grid()

    sns.kdeplot(inst.df, x = f'{model}_{inst.target}', y = f'{model}_pred', ax=ax[3])
    ax[3].scatter(inst.df[f'{model}_{inst.target}'], inst.df[f'{model}_pred'], alpha = 0.2, c = 'royalblue')
    ax[3].plot([-40,300], [-40,300], '--' , c='k')
    ax[3].grid()
    ax[3].set_title(fr"Model's prediction for {inst.time_diff} lag")
    fig.tight_layout()
    if save :
        fig.savefig(os.path.join(save_path, f'model_{model}{'_'.join(results.keys())}.png'))

def plot_wind_gust_detector(path, depid,**kwargs):
    default = {'prominence':1.5, 'height':6, 'distance':10}
    fp_params = {**default, **kwargs}
    _df = pd.read_csv(os.path.join(path, depid, f'{depid}_dive.csv'))
    data = _df['lstm'].to_numpy()
    timeframe = _df.begin_time.to_numpy()
    transition = np.diff((numpy_fill(data) >= 10).astype(int))
    first_peak = np.where((transition == 1))[0][0]
    transition[:first_peak] = 0
    left_base = np.where((transition == 1))[0]
    right_base = np.where((transition == -1))[0]
    bases = np.column_stack(
        (left_base[:np.min((len(left_base), len(right_base)))], right_base[:np.min((len(left_base), len(right_base)))]))
    wind_max, wind_mean, duration, peaks = [], [], [], []
    for base in bases:
        if base[1] - base[0] < 2:
            continue
        peaks.append(base[0] + np.argmax(np.nan_to_num(data[base[0]:base[1]])))
    peaks, _ = find_peaks(data, prominence=fp_params['prominence'], height=fp_params['height'], distance=fp_params['distance'])
    left_base, right_base = _['left_bases'], _['right_bases']
    bases = np.column_stack(
        (left_base[:np.min((len(left_base), len(right_base)))], right_base[:np.min((len(left_base), len(right_base)))]))
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(timeframe, data)
    ax.scatter(timeframe[peaks], data[peaks], c='orange', s=25)
    for base in bases:
        ax.plot([timeframe[base[0]], timeframe[base[1]]], [10, 10], '--o', c='red')
    ax.set_xlim(timeframe[0], timeframe[500])
    fig.show()