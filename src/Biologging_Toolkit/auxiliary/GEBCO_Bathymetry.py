from pykdtree.kdtree import KDTree
import numpy as np
from netCDF4 import Dataset 

def get_bathymetry(ds, bathy_path):
    bathymetry_ds = Dataset(bathy_path)
    latitude = ds['lat'][:].data
    longitude = ds['lon'][:].data
    _lat, _lon = latitude[~np.isnan(latitude)], longitude[~np.isnan(longitude)]
    tree = KDTree(bathymetry_ds['lat'][:])
    dist_lat, pos_lat = tree.query(_lat)
    tree = KDTree(bathymetry_ds['lon'][:])
    dist_lon, pos_lon = tree.query(_lon)
    bathymetry = np.full(len(latitude), np.nan)
    bathymetry[~np.isnan(latitude)] = [bathymetry_ds['elevation'][i,j] for i,j in zip(pos_lat, pos_lon)] 
    bathymetry_ds.close()
    return bathymetry