import os
from glob import glob
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import netCDF4 as nc
from tqdm import tqdm

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

def join_cfosat(dive_path = False, sens_path = False, cfosat_path = None, **kwargs) :
    default = {'time_diff':3600, 'lat_diff':0.5, 'lon_diff':0.5}
    params =  {**default, **kwargs}
    fn_cfosat = glob(os.path.join(cfosat_path, '*'))
    fn_cfosat.sort()
    time_cfosat = list(map(lambda x: get_cfosat_time(x.split('_')[-2]), fn_cfosat))
    ### FOR SES STRUCTURE
    if sens_path:
        sens = nc.Dataset(sens_path, 'a')
        sens_time = sens['time'][:].data
        sens_lat = sens['lat'][:].data
        sens_lon = sens['lon'][:].data
        indices = np.searchsorted(time_cfosat, sens_time) - 1
        indices[sens_time < time_cfosat[0]] = -5
        indices[sens_time > time_cfosat[-1]] = -5
        final_wind = np.full(len(sens_time), np.nan)
        unique_ind = np.unique(indices)
        for ind in tqdm(unique_ind):
            if ind == -5:
                rows = np.array((indices == ind).nonzero()).flatten()
                for i in rows:
                    final_wind[i] = np.nan
                continue
            rows = np.array((indices == ind).nonzero()).flatten()
            ds = nc.Dataset(fn_cfosat[ind])
            ds_time = np.array(list(map(get_nc_time, ds['row_time'][:].data)))
            ds_lat, ds_lon = ds['wvc_lat'][:].data, ds['wvc_lon'][:].data
            ds_wind = ds['wind_speed_selection'][:].data
            for i in rows:
                _wind = ds_wind
                delta_t = np.tile(abs(ds_time - sens_time[i]), ds_lat.shape[1]).reshape(ds_lat.shape)
                delta_lat = abs(ds_lat - sens_lat[i])
                delta_lon = abs(ds_lon - sens_lon[i])
                valid = (delta_t <= params['time_diff']) & (delta_lat <= params['lat_diff']) & (
                            delta_lon <= params['lon_diff'])
                weights = time_weight(delta_t, params['time_diff']) * distance_weight(delta_lat, params[
                    'lat_diff']) * distance_weight(delta_lon, params['lon_diff'])
                weighted_wind = np.nansum(_wind[valid] * weights[valid]) / np.nansum(weights[valid])
                final_wind[i] = weighted_wind
        try:
            var = sens.createVariable('cfosat', np.float32, ('time',), fill_value=np.nan)
            var_attrs = {'units': 'm/s', 'long_name':'CFOSAT cube wind value'}
            var_attrs.update(kwargs)
            for attr_name, attr_value in var_attrs.items():
                setattr(var, attr_name, attr_value)
            var[:] = final_wind
        except (FileNotFoundError, RuntimeError):
            print('No reference NetCDF file found')
        sens.close()

    ### FOR DIVE DATAFRAME
    if dive_path :
        dive_ds = pd.read_csv(dive_path)
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
        dive_ds.to_csv(dive_path, index = None)

def join_cfosat_flag(sens_path = False, cfosat_path = None, **kwargs) :
    default = {'time_diff':3600, 'lat_diff':0.5, 'lon_diff':0.5}
    params =  {**default, **kwargs}
    fn_cfosat = glob(os.path.join(cfosat_path, '*'))
    fn_cfosat.sort()
    time_cfosat = list(map(lambda x: get_cfosat_time(x.split('_')[-2]), fn_cfosat))
    ### FOR SES STRUCTURE
    if sens_path:
        sens = nc.Dataset(sens_path, 'a')
        sens_time = sens['time'][:].data
        sens_lat = sens['lat'][:].data
        sens_lon = sens['lon'][:].data
        indices = np.searchsorted(time_cfosat, sens_time) - 1
        indices[sens_time < time_cfosat[0]] = -5
        indices[sens_time > time_cfosat[-1]] = -5
        final_wind = np.full(len(sens_time), 0)
        unique_ind = np.unique(indices)
        for ind in tqdm(unique_ind):
            if ind == -5:
                rows = np.array((indices == ind).nonzero()).flatten()
                for i in rows:
                    final_wind[i] = np.nan
                continue
            rows = np.array((indices == ind).nonzero()).flatten()
            ds = nc.Dataset(fn_cfosat[ind])
            ds_time = np.array(list(map(get_nc_time, ds['row_time'][:].data)))
            ds_lat, ds_lon = ds['wvc_lat'][:].data, ds['wvc_lon'][:].data
            ds_wind = ds['wvc_quality'][:].data
            for i in rows:
                _wind = ds_wind
                delta_t = np.tile(abs(ds_time - sens_time[i]), ds_lat.shape[1]).reshape(ds_lat.shape)
                delta_lat = abs(ds_lat - sens_lat[i])
                delta_lon = abs(ds_lon - sens_lon[i])
                valid = (delta_t <= params['time_diff']) & (delta_lat <= params['lat_diff']) & (
                            delta_lon <= params['lon_diff'])
                if np.isin(_wind[valid].flatten(), [16, 2048, 2064, 4112, 6144, 6160, 4096]).all() :
                    final_wind[i] = 1
        try:
            var = sens.createVariable('cfosat_quality', np.float32, ('time',), fill_value=np.nan)
            var_attrs = {'Description':'Quality flag of data', 'Best data':'8,2048,4096'}
            var_attrs.update(kwargs)
            for attr_name, attr_value in var_attrs.items():
                setattr(var, attr_name, attr_value)
            var[:] = final_wind
        except (FileNotFoundError, RuntimeError):
            print('No reference NetCDF file found')
        sens.close()
