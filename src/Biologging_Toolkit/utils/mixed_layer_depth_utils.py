import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from scipy.signal import find_peaks, medfilt
from Biologging_Toolkit.utils.inertial_utils import coa
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter

def get_previous_wind(df, data = 'lstm') :
    if isinstance(data, list) :
        for _data in data :
            interp = interp1d(df.end_time, df[_data], bounds_error = False)
            for i in range(48):
                df[f'{_data}_{i}h'] = interp(df.end_time - i * 3600)
    else :
        interp = interp1d(df.end_time, df[data], bounds_error=False)
        for i in range(48):
            df[f'{data}_{i}h'] = interp(df.end_time - i * 3600)
    return df

def get_wind_complete(df, data = 'lstm', time_diff = 15) :
    timeframe, mld, temp, gradient, density = df.begin_time.to_numpy(), df.meop_mld.to_numpy(), df.temp10m.to_numpy(), df.gradient.to_numpy(), df.density10m.to_numpy()
    max_speed = df[data].to_numpy()
    temp_data, mld_data = [], []
    for i in range(len(timeframe)) :
        _mld, _temp = [mld[i]], [temp[i]]
        j = 1
        while (i+j < len(timeframe)) and (timeframe[i + j] - timeframe[i] < time_diff * 3600)  :
            _mld.append(mld[i + j])
            _temp.append(temp[i + j])
            j+=1
        mld_data.append(_mld)
        temp_data.append(_temp)
    return max_speed, mld_data, temp_data, gradient, density

def get_wind_gust(df, data = 'lstm', time_diff = 15, **kwargs) :
    default = {'prominence':1.5, 'height':6, 'distance':10}
    fp_params = {**default, **kwargs}
    peaks, _ = find_peaks(df[data].to_numpy(), prominence = fp_params['prominence'], height = fp_params['height'], distance = fp_params['distance'])
    timeframe, mld, temp, gradient, density = df.begin_time.to_numpy(), df.meop_mld.to_numpy(), df.temp10m.to_numpy(), df.gradient.to_numpy(), df.density10m.to_numpy()
    latitude, longitude = df.lat.to_numpy(), df.lon.to_numpy()
    begin_gust, end_gust = timeframe[_['left_bases']], timeframe[_['right_bases']]
    duration = end_gust - begin_gust
    max_speed = df[data].to_numpy()[peaks]
    average = []
    for lb, rb in zip(_['left_bases'], _['right_bases']):
        average.append(np.nanmean(df[data].to_numpy()[lb:rb+1]))
    mld_data, temp_data, other_peaks, gradient_data, density_data, distance = [], [], [], [], [], []
    for peak in peaks :
        _other_peak = [np.nan]
        _mld = [mld[peak]]
        _temp = [temp[peak]]
        gradient_data.append(gradient[peak])
        density_data.append(density[peak])
        lat1, lon1 = latitude[peak], longitude[peak]
        j = 0
        while (peak+j < len(timeframe)) and (timeframe[peak + j] - timeframe[peak] < time_diff * 3600)  :
            _mld.append(mld[peak + j])
            _temp.append(temp[peak + j])
            if (peak+j>peak) and (peak+j in peaks) :
                _other_peak.append(df[data].to_numpy()[peak+j])
            j+=1
        lat2, lon2 = latitude[peak+j-1], longitude[peak+j-1]
        distance.append(coa([lat1,lat2], [lon1, lon2]))
        other_peaks.append(len(_other_peak))
        mld_data.append(_mld)
        temp_data.append(_temp)
    return max_speed, np.array(average), duration, mld_data, np.array(other_peaks), temp_data, np.array(gradient_data), np.array(density_data), np.array(distance)

def create_gust_dataframe(depid, path, data = 'lstm', time_diff = 15, smoothing = True, structure = 'gust'):
    """
    Creates a DataFrame containing an initial state time_diff hours before MLD to be analyzed.

    This function reads dive data from a CSV file, applies optional smoothing,
    and processes it based on the chosen data structure ('gust' or 'complete').

    Parameters
    ----------
    depid : str
        Deployment ID for individual of interest.
    path : str
        File path pointing to the CSV file.
    time_diff : int, optional
        Time difference to obtain initial state in hours (default is 15).
    smoothing : bool, optional
        Whether to apply median filtering to the mixed layer depth (default is True).
    structure : str, optional
        Defines the data structure to be used, either 'gust' or 'complete' (default is 'gust').
        Gust detects wind peaks in data, complete takes all available time points.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing processed wind gust data with the following columns:
        - mld : list
            Mixed layer depth values (m).
        - peaks : list
            Detected wind gust peaks at initial state (m/s).
        - duration : list (only for 'gust' structure)
            Duration of detected gust events at initial state (s).
        - previous_mld : list
            Mixed layer depth values at initial state (m).
        - average : list (only for 'gust' structure)
            Average wind gust intensity at initial state (m/s).
        - gradient : list
            Gradient values at MLD at initial state.
        - density : list
            10m density at initial state
        - distance : list (only for 'gust' structure)
            Distance travelled by elephant seal during time_diff.
        - other_peaks : list (only for 'gust' structure)
            Additional peak gust values between initial state and analyzed MLD.
        - temp10 : list
            10m temperature depth at initial state.
        - temp_diff : list
            Temperature difference between initial state and time of analyzed MLD.
        - var_mld : list
            Variance of all mixed layer depth data between initial state and time of analyzed MLD.
        - depid : str
            Deployment ID repeated for all rows.
        - depid_id : int
            Categorical ID assigned to the deployment.
    """
    df = pd.read_csv(path)
    if smoothing :
        df['meop_mld'] = medfilt(df['meop_mld'], kernel_size=5)
    if structure == 'gust':
        peaks, average, duration, mld_data, other_peaks, temp_data, gradient, density, distance = get_wind_gust(df, data = self.data, time_diff = time_diff, **self.fp_params)
        mld, previous_mld = [_mld_data[-1] for _mld_data in mld_data], [_mld_data[0] for _mld_data in mld_data]
        df = pd.DataFrame(
            {'peaks': peaks, 'duration': duration, 'mld': mld, 'previous_mld': previous_mld, 'average': average,
             'gradient': gradient, 'density': density, 'distance': distance, 'other_peaks':other_peaks})
    elif structure == 'complete':
        peaks, mld_data, temp_data, gradient, density = get_wind_complete(df, data = data, time_diff = time_diff)
        mld, previous_mld = [_mld_data[-1] for _mld_data in mld_data], [_mld_data[0] for _mld_data in mld_data]
        df = pd.DataFrame(
            {'peaks': peaks, 'mld': mld, 'previous_mld': previous_mld,
             'gradient': gradient, 'density': density})
    df['temp10'] = [_temp_data[0] for _temp_data in temp_data]
    df['temp_diff'] = [abs(_temp_data[0] - _temp_data[-1]) for _temp_data in temp_data]
    df['var_mld'] = [np.nanvar(_mld_data, ddof=-1) for _mld_data in mld_data]
    df['depid'] = [depid]*len(df)
    df['depid_id'] = pd.Categorical(df['depid']).codes+1
    return df

def get_profiles(depid, path, data='lstm', t0=10, t1=25, filter=1, size = 40, norm=True):
    """
    Extracts and processes oceanographic profile data from a given DataFrame.

    This function retrieves mixed layer depth (MLD), wind, temperature, density, and gradient data
    from the input DataFrame, applies normalization and filtering, and interpolates the data over
    a standardized time grid.

    Parameters
    ----------
    depid : str
        Individual to get profiles from
    path : str
        Path to file containing dataframe
    data : str, optional
        Column name of the wind data to extract (default is 'lstm').
    t0 : int, optional
        Upper time bound in hours for selecting initial state data (in hours prior to the evaluated MLD) (default is 10).
    t1 : int, optional
        Lower time bound in hours for selecting initial state data (in hours prior to the evaluated MLD) (default is 25).
    filter : int, optional
        Size of the median filter applied to smooth the data (default is 1).
    size : int, optional
        Number of points in interpolation for final data output (default is 40).
    norm : bool, optional
        If True, normalizes data using min-max scaling; otherwise, keeps raw values (default is True).

    Returns
    -------
    tuple
        A tuple containing:
        - mld : numpy.ndarray
            Array of mixed layer depth values (99th quantile removed).
        - previous_mld : numpy.ndarray
            Array of previous mixed layer depth values at t1.
        - wind_data : numpy.ndarray
            Interpolated and filtered wind data over selected time interval.
        - temp_data : numpy.ndarray
            Interpolated and filtered temperature data over selected time interval.
        - density_data : numpy.ndarray
            Interpolated and filtered density data over selected time interval.
        - gradient_data : numpy.ndarray
            Interpolated and filtered gradient data over selected time interval.
    """
    df = pd.read_csv(path)
    if norm:
        normf = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    else:
        normf = lambda x: x
    timeframe, mld = df.begin_time.to_numpy(), df.meop_mld.to_numpy()
    temp, wind, gradient, density = normf(df.temp10m.to_numpy()), normf(df[data].to_numpy()), normf(df.gradient.to_numpy()), normf(df.density10m.to_numpy())
    if norm :
        lat, lon = lat_norm(df.lat.to_numpy()), lon_norm(df.lon.to_numpy())
    else :
        lat, lon = df.lat.to_numpy(), df.lon.to_numpy()
    mld[mld > np.quantile(mld, 0.99)] = np.nan
    wind_data, temp_data, gradient_data, density_data, previous_mld = [], [], [], [], []
    lat_data, lon_data, time_data = [], [], []
    for i in range(len(mld)):
        low_bound = (timeframe >= timeframe[i] - t1 * 3600)
        high_bound = (timeframe <= timeframe[i] - t0 * 3600)
        _time = timeframe[low_bound & high_bound]
        _mld = mld[low_bound & high_bound]
        previous_mld.append(_mld[0] if (~np.all(low_bound) and len(_mld) != 0) else np.nan)
        _wind = median_filter(wind[low_bound & high_bound], size=filter, mode='nearest')
        wind_data.append(interp1d(_time, _wind)(np.linspace(_time[0], _time[-1], size)) if len(_wind) != 0
                         else np.full(size, np.nan))
        _temp = median_filter(temp[low_bound & high_bound], size=filter, mode='nearest')
        temp_data.append(interp1d(_time, _temp)(np.linspace(_time[0], _time[-1], size)) if len(_temp) != 0
                         else np.full(size, np.nan))
        _density = median_filter(density[low_bound & high_bound], size=filter, mode='nearest')
        density_data.append(interp1d(_time, _density)(np.linspace(_time[0], _time[-1], size)) if len(_density) != 0
                            else np.full(size, np.nan))
        _gradient = median_filter(gradient[low_bound & high_bound], size=filter, mode='nearest')
        gradient_data.append(interp1d(_time, _gradient)(np.linspace(_time[0], _time[-1], size)) if len(_gradient) != 0
                             else np.full(size, np.nan))
        _lat = lat[low_bound & high_bound]
        lat_data.append(interp1d(_time, _lat)(np.linspace(_time[0], _time[-1], size)) if len(_temp) != 0
                         else np.full(size, np.nan))
        _lon = lon[low_bound & high_bound]
        lon_data.append(interp1d(_time, _lon)(np.linspace(_time[0], _time[-1], size)) if len(_temp) != 0
                         else np.full(size, np.nan))
        time_data.append(np.linspace(_time[0], _time[-1], size))
    return mld, np.array(previous_mld), np.array(wind_data), np.array(temp_data), np.array(density_data), np.array(gradient_data), np.array(lat_data), np.array(lon_data), np.array(time_data)

lat_norm = lambda x : (x + 90) / 180
lon_norm = lambda x : (x + 180) / 360