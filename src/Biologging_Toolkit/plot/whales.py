import os.path
from glob import glob
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
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

def pairplot(w, vars = ['jerk']) :
    pass

def plot_logistic_laws(y_pred, X_test, whale, save = False, save_path = '.'):
    regression = {'baleen':'balreg', 'delphinid':'delreg', 'spermwhale':'spermreg'}
    colors = ['deeppink', 'gold', 'magenta']
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
                ax[i,j].set_xlabel(f"Average {feature} per dive")
                ax[i,j].scatter(X_test[feature], y_pred.delphinid, c = colors[j], edgecolor = 'gray', alpha = 0.9)
                ax[i,j].grid(True)
    elif X_test.shape[1] == 1 :
        fig, ax = plt.subplots(1, 3, figsize = (15,6))
        ax[0].set_ylabel("Predicted Probability")
        feature_range = np.linspace(X_test.min(), X_test.max(), 300)
        for j, _class in enumerate(['baleen','delphinid','spermwhale']) :
            ax[j].set_title(f'{_class.capitalize()} prediction')
            probs = getattr(whale, regression[_class]).predict_proba(feature_range)[:, 1]
            ax[j].plot(feature_range, probs, c = colors[j])
            ax[j].set_xlabel(f"Average {X_test.columns[0]} per dive")
            ax[j].scatter(X_test, y_pred[_class], c = colors[j], edgecolor = 'gray', alpha = 0.9)
            ax[j].grid(True)
    fig.tight_layout()
    if save :
        fig.savefig(os.path.join(save_path, 'Logistic_regression_fits.pdf'))
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


