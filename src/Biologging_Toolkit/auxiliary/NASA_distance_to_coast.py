import numpy as np

def distance(lat, lon):
    return 6371*np.arccos(np.sin(lat[1]*np.pi/180)*np.sin(lat[0]*np.pi/180) + np.cos(lat[0]*np.pi/180)*np.cos(lat[1]*np.pi/180)*(np.cos((lon[1]-lon[0])*np.pi/180)))

def get_nearest_coast(ds, dist_path) :
    dist2shore_ds = np.loadtxt(dist_path).astype('float32')
    num_lons = int(360 / 0.04)  # Number of longitude points
    num_lats = int((180 / 0.04))  # Number of latitude points
    lon_array = np.arange(-179.98, 179.99, 0.04)
    lat_array = np.arange(-89.98, 89.99, 0.04)
    value_grid = dist2shore_ds[:,2].reshape((num_lats, num_lons, -1))
    
    latitude = ds['lat'][:].data
    longitude = ds['lon'][:].data
    shore_distance = np.full(len(latitude), np.nan)
    mask = (~np.isnan(latitude)) | (~np.isnan(longitude))
    lon_pos = np.searchsorted(lon_array, longitude[mask])
    lat_pos = num_lats - np.searchsorted(lat_array, latitude[mask])
    shore_distance[mask] = value_grid[lat_pos, lon_pos].flatten()
    return shore_distance
    
def get_nearest_shore(ds, dist_path):
    '''
    Function that computes the distance between the hydrophone and the closest shore.
    Precision is 0.04°, data is from NASA
    Make sure to call format_gps before running method
    '''
    dist2shore_ds = np.loadtxt(dist_path).astype('float32')
    def nearest_shore(x,y):
        shore_distance = np.full([len(x)], np.nan)
        x, y = np.round(x*100//4*4/100+0.02, 2), np.round(y*100//4*4/100+0.02, 2)
        _x, _y = x[(~np.isnan(x)) | (~np.isnan(y))], y[(~np.isnan(x)) | (~np.isnan(y))]
        lat_ind = np.rint(-9000/0.04*(_y)+20245500).astype(int)
        lat_ind2 = np.rint(-9000/0.04*(_y-0.04)+20245500).astype(int)
        lon_ind = (_x/0.04+4499.5).astype(int)
        sub_dist2shore = np.stack([dist2shore_ds[ind1 : ind2]  for ind1, ind2 in zip(lat_ind, lat_ind2)])
        shore_temp = np.array([sub_dist2shore[i, lon_ind[i], -1] for i in range(len(lon_ind))])
        shore_distance[(~np.isnan(x)) | (~np.isnan(y))] = shore_temp
        return shore_distance
    latitude = ds['lat'][:].data
    longitude = ds['lon'][:].data
    _lat, _lon = latitude[~np.isnan(latitude)], longitude[~np.isnan(longitude)]
    if len(np.unique(latitude)) == 1:
        shore_distance = nearest_shore(_lon[:1], _lat[:1])
        shore_distance = np.tile(shore_distance, len(ds['time'][:].data))
    else :
        shore_distance = nearest_shore(_lon, _lat)
        shore_distance = shore_distance
    return shore_distance