import os.path
from glob import glob
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


