import os
from glob import glob
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import netCDF4 as nc

get_cfosat_time = lambda x : (datetime.strptime(x, '%Y%m%dT%H%M%S').replace(tzinfo = timezone.utc).timestamp())
def get_nc_time(x) :
	try :
		return datetime.strptime(b''.join(x).decode('utf-8'), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
	except ValueError :
		return np.nan
def time_weight(delta_t, scale=3600):
    return np.exp(-abs(delta_t) / scale)
def distance_weight(delta_d, scale=1):
    return np.exp(-abs(delta_d) / scale)

def join_cfosat(path, cfosat_path, **kwargs) :
    default = {'time_diff':3600, 'lat_diff':0.5, 'lon_diff':0.5}
    params =  {**default, **kwargs}
    fn_cfosat = glob(os.path.join(cfosat_path, '*'))
    fn_cfosat.sort()
    time_cfosat = list(map(lambda x: get_cfosat_time(x.split('_')[-2]), fn_cfosat))
    dive_ds = pd.read_csv(path)
    indices = np.searchsorted(time_cfosat, dive_ds.begin_time)-1
    final_wind = np.full(len(dive_ds), np.nan)
    unique_ind = np.unique(indices)
    for ind in unique_ind :
        rows = np.array((indices == ind).nonzero()).flatten()
        ds = nc.Dataset(fn_cfosat[ind])
        ds_time = np.array(list(map(get_nc_time, ds['row_time'][:].data)))
        ds_lat, ds_lon = ds['wvc_lat'][:].data, ds['wvc_lon'][:].data
        ds_wind = ds['wind_speed_selection'][:].data
        ds_wind[ds_wind < 0] = np.nan
        for i in rows :
            _wind = ds_wind
            delta_t = np.tile(abs(ds_time - dive_ds.iloc[i].begin_time.item()), ds_lat.shape[1]).reshape(ds_lat.shape)
            delta_lat = abs(ds_lat - dive_ds.iloc[i].lat.item())
            delta_lon = abs(ds_lon - dive_ds.iloc[i].lon.item())
            valid = (delta_t <= params['time_diff']) & (delta_lat <= params['lat_diff']) & (delta_lon <= params['lon_diff'])
            weights = time_weight(delta_t, params['time_diff']) * distance_weight(delta_lat, params['lat_diff']) * distance_weight(delta_lon, params['lon_diff'])
            weighted_wind = np.nansum(_wind[valid] * weights[valid]) / np.nansum(weights[valid])
            final_wind[i] = weighted_wind
    dive_ds['cfosat_wind'] = final_wind
    dive_ds.to_csv(path, index = None)
