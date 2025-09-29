import os.path
from glob import glob
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,                # Enable LaTeX text rendering
    "font.family": "serif",             # Use a serif font
    "font.serif": ["Computer Modern"],  # Set font to Computer Modern (LaTeX default)
})
import numpy as np
import cartopy.crs as crs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from sklearn import metrics
from Biologging_Toolkit.utils.whale_utils import load_annotation_data
from Biologging_Toolkit.applications.Whales import Whales
from matplotlib.ticker import MaxNLocator
from sklearn.inspection import PartialDependenceDisplay

def histogram_detections(ker, arg, annotation_path, idx = 0):
    ker_drift, idx_ker, data_ker, arg_drift, idx_arg, data_arg = load_annotation_data(ker, arg, annotation_path)
    data_arg = data_arg[idx_arg >= idx]
    data_ker = data_ker[idx_ker >= idx]
    _label_arg, _count_arg = np.unique(data_arg, return_counts=True)
    _label_ker, _count_ker = np.unique(data_ker, return_counts=True)
    hist_arg = pd.DataFrame({'Number of drift dives':len(data_arg),
                            'Baleen':_count_arg[np.isin(_label_arg, ['ABW','FW','MW','SW','HW','Sweep','Downsweep','SRW'])].sum(),
                            'Other odontocetes':_count_arg[np.isin(_label_arg, ['Unidentified clicks', 'Buzz', 'Clicks', 'Delphinid clicks', 'Delphinid whistle'])].sum(),
                            'Spermwhale':_count_arg[np.isin(_label_arg, ['Spermwhale'])]}).T.reset_index()
    hist_arg.columns = ['Label', 'Count']
    hist_ker = pd.DataFrame({'Number of drift dives':len(data_ker),
                            'Baleen':_count_ker[np.isin(_label_ker, ['ABW','FW','MW','SW','HW','Sweep','Downsweep','SRW'])].sum(),
                            'Other odontocetes':_count_ker[np.isin(_label_ker, ['Unidentified clicks', 'Buzz', 'Clicks', 'Delphinid clicks', 'Delphinid whistle'])].sum(),
                            'Spermwhale':_count_ker[np.isin(_label_ker, ['Spermwhale'])]}).T.reset_index()
    hist_ker.columns = ['Label', 'Count']
    primary_label = 'Number of drift dives'

    primary_arg = hist_arg[hist_arg['Label'] == primary_label]
    secondary_arg = hist_arg[hist_arg['Label'] != primary_label]
    primary_ker = hist_ker[hist_ker['Label'] == primary_label]
    secondary_ker = hist_ker[hist_ker['Label'] != primary_label]

    primary_max = max(primary_arg['Count'].max(), primary_ker['Count'].max())
    secondary_max = max(secondary_arg['Count'].max(), secondary_ker['Count'].max())

    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=False)

    # Argentina subplot
    ax1 = axs[0]
    ax1.bar(primary_arg['Label'], primary_arg['Count'], color='tab:blue')
    ax1.set_ylabel('Number of drift dives', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, primary_max * 1.1)
    ax1_2 = ax1.twinx()
    ax1_2.bar(secondary_arg['Label'], secondary_arg['Count'], color='tab:orange')
    ax1_2.set_ylabel('Other counts', color='tab:orange')
    ax1_2.tick_params(axis='y', labelcolor='tab:orange')
    ax1_2.set_ylim(0, secondary_max * 1.1)
    ax1.set_title('Argentina')
    ax1.set_xticklabels(['Number of drift dives', 'Baleen', 'Other odontocetes', 'Spermwhale'], rotation=45)

    # Kerguelen subplot
    ax2 = axs[1]
    ax2.bar(primary_ker['Label'], primary_ker['Count'], color='tab:blue')
    ax2.set_ylabel('Number of drift dives', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, primary_max * 1.1)
    ax2_2 = ax2.twinx()
    ax2_2.bar(secondary_ker['Label'], secondary_ker['Count'], color='tab:orange')
    ax2_2.set_ylabel('Other counts', color='tab:orange')
    ax2_2.tick_params(axis='y', labelcolor='tab:orange')
    ax2_2.set_ylim(0, secondary_max * 1.1)
    ax2.set_title('Kerguelen')
    ax2.set_xticklabels(['Number of drift dives', 'Baleen', 'Other odontocetes', 'Spermwhale'], rotation=45)

    plt.tight_layout()
    plt.show()

def plot_confidence_index_detections(ker, arg, annotation_path, results_path, save = False, save_path = '.') :
    w = Whales(ker + arg,
               annotation_path=[os.path.join(annotation_path, dep, 'formatted_timestamps.csv') for dep in ker + arg])
    results = pd.DataFrame()
    for i in [0, 1, 2]:
        w.idx = i
        w.load_data(annotation_path=os.path.join(results_path, f'annotations_{i}.pkl'),
                    daily_path=os.path.join(results_path, f'daily_pool_{i}.pkl'))
        df = pd.concat(w.annotations.values(), ignore_index=True)
        df['Confidence Index'] = i
        _df = df[['baleen', 'spermwhale', 'delphinid', 'Confidence Index']]
        _df.columns = ['Baleen', 'Spermwhale', 'Delphinid', 'Confidence Index']
        results = pd.concat((results, _df), ignore_index=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    results = results.melt(id_vars='Confidence Index')
    results.columns = ['Confidence Index', 'Annotation', 'Number of drift dives with detections']
    custom_colors = ['darkblue', 'yellow','turquoise']
    agg_results = results.groupby(['Confidence Index', 'Annotation'], as_index=False)[
        'Number of drift dives with detections'].sum()
    sns.barplot(data=agg_results, x='Confidence Index', y='Number of drift dives with detections', hue='Annotation',
                ax=ax, palette=custom_colors, alpha = 1)
    ax.grid(False)
    print(agg_results)
    if save :
        fig.savefig(save_path)

def score_confidence_index(score_path, save = False, save_path = '.') :
    df = pd.read_csv(score_path, header=None)
    df1 = pd.DataFrame()
    df1['F1'] = df.iloc[2:, [3, 6, 9, 12, 15]].astype(float).mean(axis=1)
    df1['Recall'] = df.iloc[2:, [4, 7, 10, 13, 16]].astype(float).mean(axis=1)
    df1['Precision'] = df.iloc[2:, [5, 8, 11, 14, 17]].astype(float).mean(axis=1)
    df1['Specie'] = df[0].iloc[2:]
    df1['Confidence'] = df[1].iloc[2:]
    df1['Population'] = df[2].iloc[2:]

    specie_name = {"Baleen":"Baleen whales", "Delphinid":"Delphinids","Spermwhale":"Spermwhale"}
    df_long = df1.melt(id_vars=["Population", "Specie", "Confidence"],value_vars=["F1", "Recall", "Precision"],var_name="metric",value_name="score")
    conf_order = df_long["Confidence"].unique()
    conf_positions = {sp: i * 2 + 1 for i, sp in enumerate(conf_order)}
    pop_offsets = {p: -0.2 if i == 0 else 0.2 for i, p in enumerate(df_long["Population"].unique())}
    pop_symbol = {'Arg': '^', 'Ker': 's'}
    pop_name = {"Arg": "Argentina", "Ker": "Kerguelen"}
    metric_colors = {"F1": "red", "Recall": "blue", "Precision": "green"}
    species_levels = df_long["Specie"].unique()
    with plt.rc_context({'font.size': 14}):
        fig, axes = plt.subplots(1, len(species_levels), figsize=(18, 6), sharey=True)
        for ax, spec in zip(axes, species_levels):
            df_sub = df_long[df_long["Specie"] == spec]
            for _, row in df_sub.iterrows():
                x_base = conf_positions[row["Confidence"]]
                x = x_base + pop_offsets[row["Population"]]
                ax.scatter(x, row["score"],color=metric_colors[row["metric"]],marker=pop_symbol[row['Population']], s=100, edgecolor="black")
                ax.set_xticks(list(conf_positions.values()))
            ax.set_xticklabels([''.join([x[0].upper() for x in value.split(' ')]) for value in list(conf_positions.keys())])
            ax.set_title(f"{specie_name[spec]}")
            ax.set_xlabel("")
            ax.set_ylabel("Score")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        handles = [plt.Line2D([0], [0], marker="s", color=c, linestyle="", markersize=10, markeredgecolor="black")
                   for m, c in metric_colors.items()]
        labels = list(metric_colors.keys())
        pop_handles = [plt.Line2D([0], [0], marker=mk, color="gray", linestyle="", markersize=10, markeredgecolor="black")
            for p, mk in pop_symbol.items()]
        pop_labels = [pop_name[key] for key in pop_symbol.keys()]
        fig.legend(handles=handles + pop_handles,labels=labels + pop_labels, bbox_to_anchor=(0.9, 0.6),loc="upper left",frameon=True)
        if save :
            fig.savefig(os.path.join(save_path, 'score_confidence_index.pdf'), bbox_inches='tight')
def pairplot(w, vars = ['jerk'], arg = [], ker = [], save = False, save_path = '.') :
    names = {'surface_temp':'Surface temperature', 'bathy':'Bathymetry', 'flash':'Flashes',
             'jerk':'PrCAs', 'temp':'Temperature', 'sal':'Salinity', 'loc':'Population', 'chla_ctd':'Chlorophyll-a'}
    df = pd.DataFrame()
    for key in w.daily.keys() :
        if key in arg :
            w.daily[key]['loc'] = 'Argentina'
        elif key in ker :
            w.daily[key]['loc'] = 'Kerguelen'
        else :
            w.daily[key]['loc'] = 'any'
        df = pd.concat((df, w.daily[key]))
    df.reset_index(inplace = True)
    df.columns = [names[_col] if _col in list(names.keys()) else _col for _col in list(df.columns)]
    vars.append('loc')
    vars = [names[var] for var in vars]
    g = sns.pairplot(df[vars], hue = 'Population')
    if save :
        g.tight_layout()
        g.savefig(os.path.join(save_path, 'var_pairplot.pdf'))

def plot_partial_laws(y_pred, X_test, whale, group = 'baleen', save = False, save_path = '.') :
    regression = {'baleen':'balreg', 'delphinid':'delreg', 'spermwhale':'spermreg'}
    colors = {'baleen':'darkblue', 'delphinid':'gold', 'spermwhale':'turquoise'}
    features = {'bathy':'bathymetry (m)','jerk':'number of PrCAs', 'flash':'number of flashes','temp':'temperature (°C)',
                'surface_temp':'surface temperature (°C)', 'temp_ctd':'temperature (°C)', 'chla_ctd':r'Chlorophyll-a ($mg/m^3$)',
                'sal_ctd':'salinity (PSU)', 'surface_temperature':'surface temperature (°C)',
                'salinity':'salinity (PSU)', 'temperature':'temperature (°C)', 'surface_salinity':'surface salinity (PSU)'}
    variables = list(X_test.columns)
    fig, ax = plt.subplots(1, len(variables), figsize = (3*len(variables), 3.3))
    for i, feature in enumerate(variables) :
        feature_range = np.linspace(X_test[feature].min(), X_test[feature].max(), 300)
        med = X_test.loc[:, variables[:i] + variables[i+1:]].median()
        X_plot = pd.DataFrame({**{feature: feature_range}, **{_var:np.full(300, float(med[_var])) for _var in variables[:i]+variables[i+1:]}})
        X_plot = X_plot[variables]
        probs = getattr(whale, regression[group]).predict_proba(X_plot)
        ax[i].plot(feature_range, probs[:, 1], c=colors[group], linewidth = 2)
        ax[i].set_xlabel(f"Average {features[feature]}\nper dive", fontsize=14)
        ax[i].xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax[i].tick_params(axis='both', labelsize=12)
        #ax[i].scatter(X_test[feature], y_pred[group], c = colors[group], edgecolor = 'gray', alpha = 0.9)
        ax[i].grid(True)
    ax[0].set_ylabel("Predicted Probability", fontsize=14)
    fig.tight_layout()
    if save :
        fig.savefig(os.path.join(save_path, f'partial_{group}.pdf'), bbox_inches='tight', pad_inches=0.3)
    fig.show()

def plot_logistic_laws(y_pred, X_test, whale, save = False, save_path = '.'):
    regression = {'baleen':'balreg', 'delphinid':'delreg', 'spermwhale':'spermreg'}
    colors = ['deeppink', 'gold', 'magenta']
    features = {'bathy':'bathymetry','jerk':'number of PrCAs', 'flash':'number of flashes','temp':'temperature',
                'surface_temp':'surface temperature', 'temp_ctd':'temperature', 'chla_ctd':'Chlorophyll-a',
                'sal_ctd':'salinity', 'surface_temperature':'surface temperature'}
    if X_test.shape[1] == 2 :
        fig, ax = plt.subplots(2, 3, figsize = (15,11))
        for i, (feature, fixed) in enumerate(zip(['flash','jerk'],['jerk','flash'])) :
            ax[i, 0].set_ylabel("Predicted Probability")
            feature_range = np.linspace(X_test[feature].min(), X_test[feature].max(), 300)
            fixed_mean = X_test[fixed].mean()
            X_plot = pd.DataFrame({'jerk': feature_range, 'flash': fixed_mean})
            for j, _class in enumerate(['baleen','delphinid','spermwhale']) :
                if i == 0 :
                    ax[i, j].set_title(f'{_class.capitalize()} prediction')
                probs = getattr(whale, regression[_class]).predict_proba(X_plot)[:, 1]
                ax[i,j].plot(feature_range, probs, c = colors[j])
                ax[i,j].set_xlabel(f"Average {features[feature]} per dive")
                ax[i,j].scatter(X_test[feature], y_pred.delphinid, c = colors[j], edgecolor = 'gray', alpha = 0.9)
                ax[i,j].grid(True)
    elif X_test.shape[1] == 1 :
        fig, ax = plt.subplots(1, 3, figsize = (15,6))
        ax[0].set_ylabel("Predicted Probability")
        feature_range = np.linspace(X_test.min(), X_test.max(), 300)
        for j, (_class, _class_name) in enumerate(zip(['baleen','delphinid','spermwhale'], ['baleen whale','delphinid','spermwhale'])) :
            ax[j].set_title(f'{_class_name.capitalize()} prediction')
            probs = getattr(whale, regression[_class]).predict_proba(feature_range)[:, 1]
            ax[j].plot(feature_range, probs, c = colors[j])
            ax[j].set_xlabel(f"Average {features[X_test.columns[0]]} per dive")
            ax[j].scatter(X_test, y_pred[_class], c = colors[j], edgecolor = 'gray', alpha = 0.9)
            ax[j].grid(True)
    fig.tight_layout()
    if save :
        fig.savefig(os.path.join(save_path, 'Logistic_regression_fits.pdf'))
    fig.show()

def combined_logistic_laws(y_pred, X_test, whale, save = False, save_path = '.'):
    regression = {'baleen':'balreg', 'delphinid':'delreg', 'spermwhale':'spermreg'}
    #colors = ['deeppink', 'gold', 'navy']
    features = {'bathy':'bathymetry','jerk':'number of PrCAs', 'flash':'number of flashes','temp':'temperature',
                'surface_temp':'surface temperature', 'temp_ctd':'temperature', 'chla_ctd':'chlorophyll-a',
                'sal_ctd':'salinity', 'surface_temperature':'surface temperature', 'salinity':'salinity',
                'surface_salinity':'surface salinity'}
    colors = ['turquoise','navy', 'gold']
    if X_test.shape[1] == 2 :
        fig, ax = plt.subplots(2, 3, figsize = (15,11))
        for i, (feature, fixed) in enumerate(zip(['flash','jerk'],['jerk','flash'])) :
            ax[i, 0].set_ylabel("Predicted Probability")
            feature_range = np.linspace(X_test[feature].min(), X_test[feature].max(), 300)
            fixed_mean = X_test[fixed].mean()
            X_plot = pd.DataFrame({'jerk': feature_range, 'flash': fixed_mean})
            for j, _class in enumerate(['baleen','delphinid','spermwhale']) :
                if i == 0 :
                    ax[i, j].set_title(f'{_class.capitalize()} prediction')
                probs = getattr(whale, regression[_class]).predict_proba(X_plot)[:, 1]
                ax[i,j].plot(feature_range, probs, c = colors[j])
                ax[i,j].set_xlabel(f"Average {features[feature]} per dive")
                ax[i,j].scatter(X_test[feature], y_pred.delphinid, c = colors[j], edgecolor = 'gray', alpha = 0.9)
                ax[i,j].grid(True)
    elif X_test.shape[1] == 1 :
        fig, ax = plt.subplots(figsize = (4,4))
        ax.set_ylabel("Predicted Probability")
        feature_range = np.linspace(X_test.min(), X_test.max(), 300)
        for j, (_class, _class_name) in enumerate(zip(['spermwhale','baleen', 'delphinid'], ['spermwhale','baleen whale','delphinid'])) :
            probs = getattr(whale, regression[_class]).predict_proba(feature_range)[:, 1]
            ax.plot(feature_range, probs, c = colors[j])
            ax.set_xlabel(f"Average {features[X_test.columns[0]]} per dive")
            ax.scatter(X_test, y_pred[_class] + (j-1)*0.019, c = colors[j], edgecolor = 'black', linewidth=0.2, alpha = 0.7)
            ax.grid(True)
    fig.tight_layout()
    if save :
        fig.savefig(os.path.join(save_path, 'combined_regression_fits.pdf'))
    fig.show()

def prediction_conf_matrix(y_test, y_pred, save = False, save_path = '.', type = 'logistic'):
    fig, ax = plt.subplots(1,3, figsize = (15, 5))
    class_names = [0, 1]
    for i, _class in enumerate(['baleen','spermwhale','delphinid']) :
        ax[i].set_title(f'{_class.capitalize()} prediction')
        cnf_matrix = metrics.confusion_matrix(y_test[_class], y_pred[_class], normalize = 'true')
        tick_marks = np.arange(len(class_names))
        ax[i].set_xticks(tick_marks, class_names)
        ax[i].set_yticks(tick_marks, class_names)
        sns.heatmap(pd.DataFrame(cnf_matrix), vmin = 0, vmax = 1, annot=True,
                    cbar = False, square = True, cmap="YlGnBu", ax = ax[i])
        ax[i].set_xlabel('Predicted label')
    ax[0].set_ylabel('Actual label')
    fig.subplots_adjust(right=0.85, left=0.05, bottom=0.05, top=0.95, wspace=0.08)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label='Normalized accuracy')
    fig.show()
    if save :
        fig.savefig(os.path.join(save_path, f'{type}_matrix_whales.pdf'))


def patron_saisonnier(w, ker, arg):

    final_labels = ['SW', 'HW', 'SRW', 'Spermwhale', 'FW', 'MW', 'ABW', 'Delphinid']
    fig = plt.figure(figsize=(15, 25))
    gs0 = gridspec.GridSpec(4, 2, figure=fig)

    axes = []
    for i in range(8):
        inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[i])
        sub_axes = []
        ax0 = fig.add_subplot(inner[0])
        ax0.set_title(final_labels[i])
        ax0.text(1.02, 0.5, "Kerguelen",transform=ax0.transAxes,rotation=-90, va="center", ha="left")
        sub_axes.append(ax0)
        ax1 = fig.add_subplot(inner[1])
        ax1.text(1.02, 0.5, "Peninsula Valdés",transform=ax1.transAxes,rotation=-90, va="center", ha="left")
        sub_axes.append(ax1)
        inner_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=inner[2], wspace=0.3)
        ax2_left = fig.add_subplot(inner_inner[0])
        ax2_right = fig.add_subplot(inner_inner[1])
        sub_axes.extend([ax2_left, ax2_right])
        axes.append(sub_axes)

    for i, population in enumerate([ker, arg]) :
        df = pd.concat([w.annotations[_ml] for _ml in population])
        annotation_cols = ["Annotation", "Annotation2", "Annotation3"]
        unique_labels = ['SW', 'HW', 'SRW', 'Spermwhale', 'FW', 'MW', 'ABW', 'Buzz', 'Clicks', 'Delphinid clicks', 'Delphinid whistle']
        for label in unique_labels:
            df[label] = df[annotation_cols].eq(label).any(axis=1).astype(int)
        df['Delphinid'] = np.max(df[['Clicks', 'Delphinid clicks', 'Delphinid whistle', 'Buzz']].to_numpy(), axis = 1)
        daily = df[['date','duration', 'lat','lon']+final_labels].groupby('date').agg({'duration':'sum','lat':'mean','lon':'mean',})
        for label in final_labels:
            _temp = df[['date','duration']][df[label] == 1].groupby('date').agg('sum')
            daily = daily.join(_temp, on='date', rsuffix=f'_{label}').fillna({f'duration_{label}': 0})
        daily.reset_index(inplace = True)
        daily["date"] = pd.to_datetime(daily.date)
        daily["month_day"] = daily["date"].dt.strftime("%m-%d")
        daily = daily.groupby("month_day")[["duration", "duration_SW", "duration_Delphinid", "duration_Spermwhale", "duration_HW",
                                    "duration_ABW", "duration_SRW", "duration_MW", "duration_FW"]].sum().reset_index()
        daily.loc[0] = ['10-08',1,0,0,0,0,0,0,0,0]
        daily.loc[len(daily)] = ['12-16',1,0,0,0,0,0,0,0,0]
        daily["month_day_dt"] = pd.to_datetime("2021-" + daily["month_day"])
        for j, label in enumerate(final_labels) :
            axes[j][i].bar(daily.month_day_dt, daily[f'duration_{label}']/daily.duration)
            xticks = pd.to_datetime(["2021-10-15", "2021-11-01", "2021-11-15", "2021-12-01", "2021-12-15"])
            if i == 0:
                axes[j][i].set_xticks(xticks)
                axes[j][i].set_xticklabels(["","","","",""])
            if i == 1:
                axes[j][i].set_xticks(xticks)
                axes[j][i].set_xticklabels([d.strftime("%b %d") for d in xticks])
            axes[j][i].set_ylim(0,1)
            axes[j][i].set_yticks([0, 0.25, 0.5, 0.75, 1])


def bubble_map(whale, depid, colorbar = True, legend_loc = 'upper left', save = True, save_path = '.'):
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': crs.PlateCarree()})
    ax = ax.flatten()
    if 'pos' in dir(whale) :
        lat1, lat2 = whale.pos[depid].lat.min() - 5, whale.pos[depid].lat.max() + 5
        lon1, lon2 = whale.pos[depid].lon.min() - 5, whale.pos[depid].lon.max() + 5
    else :
        lat1, lat2 = whale.pos[depid].lat.min() - 5, whale.pos[depid].lat.max() + 5
        lon1, lon2 = whale.pos[depid].lon.min() - 5, whale.pos[depid].lon.max() + 5
    for i in range(4):
        ax[i].set_extent([lon1, lon2, lat2, lat1], crs=crs.PlateCarree())
        #ax[i].stock_img()
        depths_str, shp_dict, colors_depths, blues_cm = load_bathymetry()
        for j, depth_str in enumerate(depths_str):
            ax[i].add_geometries(shp_dict[depth_str].geometries(),
                              crs=crs.PlateCarree(),
                              color=colors_depths[j])
        ax[i].add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax[i].add_feature(cfeature.LAND, edgecolor='black')
        ax[i].add_feature(cfeature.COASTLINE, linewidth = 0.5)
        ax[i].add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
        gl = ax[i].gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (i % 2 == 0)
        gl.bottom_labels = (i // 2 == 1)
        gl.xlines = False
        gl.ylines = False
        if 'pos' in dir(whale) :
            ax[i].plot(whale.pos[depid].lon, whale.pos[depid].lat, c = 'k')
        else :
            ax[i].plot(whale.annotations[depid].lon, whale.annotations[depid].lat, c = 'k')
    ax[0].scatter(whale.daily[depid].lon, whale.daily[depid].lat,
                  s = whale.daily[depid].jerk*20, zorder = 10,
                  c = 'firebrick', edgecolors = 'grey', alpha = 0.8, label = 'PrCA')
    prca_levels = [1,4,7,10]
    handles = [
        Line2D([], [], marker='o', color='w',
               markerfacecolor='firebrick', markeredgecolor='grey',
               markersize=(f * 20) ** 0.5, label=f'{f} PrCAs') for f in prca_levels]
    ax[0].legend(handles=handles, title='PrCA', loc=legend_loc)
    ax[1].scatter(whale.daily[depid].lon, whale.daily[depid].lat,
                  s = whale.daily[depid].flash*5, zorder = 10,
                  c = 'orange', edgecolors = 'grey', alpha = 0.8, label = 'Flash')
    flash_levels = [10, 25, 40, 55]
    handles = [
        Line2D([], [], marker='o', color='w',
               markerfacecolor='orange', markeredgecolor='grey',
               markersize=(f * 5) ** 0.5, label=f'{f} flashes') for f in flash_levels]
    ax[1].legend(handles=handles, title='Flashes', loc=legend_loc)
    ax[2].scatter(whale.daily[depid].lon, whale.daily[depid].lat,
                  s = whale.daily[depid].duration_baleen / whale.daily[depid].duration * 200, zorder = 10,
                  c = 'deeppink', edgecolors = 'grey', alpha = 0.8, label = 'Baleen whales')
    first_legend = ax[2].legend(loc='upper left' if legend_loc != 'upper left' else 'lower left')
    detection_levels = [10,25,50,75]
    handles = [
        Line2D([], [], marker='o', color='w',
               markerfacecolor='lightgrey', markeredgecolor='grey',
               markersize=(f * 2) ** 0.5, label=f'{f} %') for f in detection_levels]
    second_legend = ax[2].legend(handles=handles, title='Positive minute ratio', loc=legend_loc)
    ax[2].add_artist(first_legend)
    ax[3].scatter(whale.daily[depid].lon, whale.daily[depid].lat,
                  s = whale.daily[depid].duration_delphinid / whale.daily[depid].duration * 200, zorder = 10,
                  c = 'gold', edgecolors = 'grey', alpha = 0.8, label = 'Delphinid')
    ax[3].scatter(whale.daily[depid].lon, whale.daily[depid].lat,
                  s = whale.daily[depid].duration_spermwhale / whale.daily[depid].duration * 200, zorder = 10,
                  c = 'magenta', edgecolors = 'grey', alpha = 0.8, label = 'Spermwhale')
    ax[3].legend()
    ax[0].set_rasterized(True)
    ax[1].set_rasterized(True)
    ax[2].set_rasterized(True)
    ax[3].set_rasterized(True)
    if colorbar :
        axi = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        fig.subplots_adjust(right=0.85, left = 0.05, bottom = 0.05, top = 0.95, wspace = 0.08, hspace = 0.01)
        depths = depths_str.astype(int)
        nudge = 0.01
        N = len(depths)
        boundaries = [min(depths)] + sorted(depths + nudge)
        norm = matplotlib.colors.BoundaryNorm(boundaries, N)
        sm = plt.cm.ScalarMappable(cmap=blues_cm, norm=norm)
        fig.colorbar(mappable=sm,
                     cax=axi,
                     spacing='proportional',
                     extend='min',
                     ticks=depths,
                     label='Depth (m)')
    if save :
        fig.savefig(save_path)
    fig.show()

def load_bathymetry()  :  # Load data (14.8 MB file)
    depths_str, shp_dict = download_bathymetry(
        'https://naturalearth.s3.amazonaws.com/' +
        '10m_physical/ne_10m_bathymetry_all.zip')
        # Construct a discrete colormap with colors corresponding to each depth
    depths = depths_str.astype(int)
    N = len(depths)
    nudge = 0.01  # shift bin edge slightly to include data
    boundaries = [min(depths)] + sorted(depths+nudge)  # low to high
    norm = matplotlib.colors.BoundaryNorm(boundaries, N)
    blues_cm = matplotlib.colormaps['Blues_r'].resampled(N)
    colors_depths = blues_cm(norm(depths))
    return depths_str, shp_dict, colors_depths, blues_cm


'''
    sm = plt.cm.ScalarMappable(cmap=blues_cm, norm=norm)
    fig.colorbar(mappable=sm,
                 cax=axi,
                 spacing='proportional',
                 extend='min',
                 ticks=depths,
                 label='Depth (m)')

    # Convert vector bathymetries to raster (saves a lot of disk space)
    # while leaving labels as vectors'''

def download_bathymetry(zip_file_url):
    """Read zip file from Natural Earth containing bathymetry shapefiles"""
    # Download and extract shapefiles
    import io
    import zipfile

    import requests
    if os.path.exists('ne_10m_bahymetry_all') :
        pass
    else :
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("ne_10m_bathymetry_all/")

    # Read shapefiles, sorted by depth
    shp_dict = {}
    files = glob('ne_10m_bathymetry_all/*.shp')
    assert len(files) > 0
    files.sort()
    depths = []
    for f in files:
        depth = '-' + f.split('_')[-1].split('.')[0]  # depth from file name
        depths.append(depth)
        bbox = (90, -15, 160, 60)  # (x0, y0, x1, y1)
        nei = shpreader.Reader(f, bbox=bbox)
        shp_dict[depth] = nei
    depths = np.array(depths)[::-1]  # sort from surface to bottom
    return depths, shp_dict


