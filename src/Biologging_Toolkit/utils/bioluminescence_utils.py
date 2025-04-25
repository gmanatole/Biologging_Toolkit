import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import numpy as np
import calendar
import time
import pandas as pd
import netCDF4 as nc
from Biologging_Toolkit.config.config import *

def get_start_time_sens(x) :
	"""
	Turn string UTC datetime (from sens5 structure) into a POSIX timestamp (UTC)
	"""
	return datetime.strptime(x, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()

def get_time_trk(ds) :
	timestamp = get_start_time_sens(ds.dephist_device_datetime_start) + ds['POS'][:].data[0]
	if min(timestamp) > datetime(2022,1,1).timestamp() :
		timestamp = []
		for x in ds['POS'][:].data :
			timestamp.append(calendar.timegm((datetime.fromordinal(int(x)) + timedelta(days=x % 1) - timedelta(days=366)).timetuple()))
		timestamp = np.array(timestamp)
	return timestamp

def get_ext_time_xml(x) :
	"""
	Turn string UTC datetime (from xml structure) into a POSIX timestamp (UTC)
	"""	
	return datetime.strptime(x, '%Y,%m,%d,%H,%M,%S').replace(tzinfo=timezone.utc).timestamp()

def get_start_time_xml(path) :
	"""
	Looks in .xml file for keyword ACC and returns associated TIME as a POSIX timestamp.
	Code assumes that TIME is in UTC and returns UTC timestamp
	"""
	if isinstance(path, (str, np.str_)):
		tree = ET.parse(path)
		root = tree.getroot()
		for cfg in root.findall('CFG'):
		    if cfg.find('RANGE[@SENS="ACC"]') != None:
		        return datetime.strptime(cfg.attrib.get('TIME'), '%Y,%m,%d,%H,%M,%S').replace(tzinfo=timezone.utc).timestamp()
		raise ValueError(f'starting time not found in xml file {path}')

	if isinstance(path, (list, np.ndarray)) :
		xml_dates = []
		for i, fn in enumerate(path) :
			try :
				tree = ET.parse(fn)
				root = tree.getroot()
				for cfg in root.findall('CFG'):
					if cfg.find('RANGE[@SENS="ACC"]') != None :
						xml_dates.append(datetime.strptime(cfg.attrib.get('TIME'), '%Y,%m,%d,%H,%M,%S').replace(tzinfo=timezone.utc).timestamp())
				if i+1 != len(xml_dates) :
					print(f'starting time not found in xml file {path[i]}')
			except ET.ParseError :
				continue
		return np.array(xml_dates)
	
	
def get_start_date_xml(path) :
	get_xml_date = lambda x : calendar.timegm(time.strptime(x, '<EVENT TIME="%Y,%m,%d,%H,%M,%S">'))
	if isinstance(path, (str, np.str_)):
		with open(path) as f:
			line = f.readline().strip()
			while line.find('EVENT TIME') == -1:
				line = f.readline().strip()
		return get_xml_date(line)
	
	if isinstance(path, (list, np.ndarray)) :
		xml_dates = []
		for i, fn in enumerate(path) :
			with open(fn) as f:
				line = f.readline().strip()
				while line.find('EVENT TIME') == -1:
					line = f.readline().strip()
			xml_dates.append(get_xml_date(line))
		return np.array(xml_dates)
		

def get_xml_columns(path, **kwargs) :
	"""
	Enter a xml path and dictionary containing keys (must be column name from d3sensordefs file) and values for those keys
	For instance add as argument cal='acc', qualifier2='d4' if you want DTAG4 accelerometer data.
	"""
	ds = pd.read_csv(SES_PATH.xml_data_columns)
	values = {**kwargs}
	mask = np.all([ds[key] == values[key] for key in values.keys()], axis=0)
	num_name = ds['number'].to_numpy()[mask]
	tree = ET.parse(path)
	root = tree.getroot()
	chans_element = root.find(".//CHANS[@N='18']")
	if chans_element is not None:
		chans_numbers = list(map(int, chans_element.text.split(',')))
		chans_array = np.array(chans_numbers)
	else :
		return 'Column number names were not found'
	return np.squeeze([np.where(chans_array == num)[0] for num in num_name if np.any(chans_array == num)])
	
	
	
def get_boundaries_metadata(path) :
	"""
	Enter path to built data structure to get time and space boundaries.
	This data can then be used to download ERA
	"""
	ds = nc.Dataset(path)
	time_min = datetime.fromtimestamp(np.nanmin(ds['time'][:])).replace(tzinfo=timezone.utc).replace(minute = 0, second = 0)
	time_max = datetime.fromtimestamp(np.nanmax(ds['time'][:])).replace(tzinfo=timezone.utc)
	time_max = time_max.replace(hour=time_max.hour+1, minute = 0, second = 0)
	print('Dataset begins at ', time_min)
	print('and ends at       ', time_max)
	lat_min =  np.floor(np.nanmin(ds['lat'][:]) * 4) / 4
	lat_max =  np.ceil(np.nanmax(ds['lat'][:]) * 4) / 4
	lon_min =  np.floor(np.nanmin(ds['lon'][:]) * 4) / 4	
	lon_max =  np.ceil(np.nanmax(ds['lon'][:]) * 4) / 4
	print('The GPS boundaries are :')
	for bound, card in zip([lat_min, lat_max, lon_min, lon_max], ['South', 'North', 'West', 'East']):
		print(f'    - {card:<6} : {bound: 0.2f}°')
	
def to_timestamp(string: str) -> datetime:
    if isinstance(string, datetime):
        return string
    try:
        return datetime.strptime(string, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        try:
            return datetime.strptime(string, "%Y-%m-%dT%H-%M-%S_%fZ")
        except ValueError:
            try:
                return datetime.strptime(string, "%Y-%m-%dT%H:%M:%S.%f%z")
            except ValueError:
                try:
                    return datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    raise ValueError(
                        f"The timestamp '{string}' must match either format %Y-%m-%dT%H:%M:%S.%fZ or %Y-%m-%dT%H-%M-%S_%fZ"
                    )

def get_epoch(df):
    "Function that adds epoch column to dataframe"
    df["timestamp"] = df.timestamp.apply(lambda x: to_timestamp(x)).dt.tz_localize(None)
    if "epoch" in df.columns:
        return df
    try:
        df["epoch"] = df.timestamp.apply(lambda x: x.timestamp())
    except ValueError:
        print(
            "Please check that you have either a timestamp column (format ISO 8601 Micro s) or an epoch column"
        )
    df["timestamp"] = df.timestamp.apply(from_timestamp)
    return df

def from_timestamp(date: datetime) -> str:
    return datetime.strftime(date, "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
	
	
def resample_boolean_array(arr, N):
    # Create an array of evenly spaced indices over the new size N
    original_indices = np.linspace(0, len(arr) - 1, N)
    # Round the indices to nearest integers and cast to integers
    resampled_indices = np.round(original_indices).astype(int)
    # Use these indices to pick elements from the original array
    resampled_arr = [arr[i] for i in resampled_indices]
    return resampled_arr
	
def get_train_test_split(paths, indices, depids, method = 'random_split', test_depid = None, split = 0.8) :
	if method == 'depid' :
		if isinstance(test_depid, str) :
			test_depid = [test_depid]
		training = np.where(~np.isin(depids, test_depid))[0]
		testing = np.where(np.isin(depids, test_depid))[0]
		return (paths[training], indices[training]), (paths[testing], indices[testing])

	elif method == 'temporal_split' :
		return (paths[:int(split * len(indices))], indices[:int(split * len(indices))]), (paths[int(split * len(indices)):], indices[int(split * len(indices)):])

	elif method == 'random_split' :
		np.random.seed(32)
		suffle_idx = list(range(len(indices)))
		random.shuffle(suffle_idx)
		indices = [indices[i] for i in suffle_idx]
		paths = [paths[i] for i in suffle_idx]
		return (paths[:int(split * len(indices))], indices[:int(split * len(indices))]), (paths[int(split * len(indices)):], indices[int(split * len(indices)):])
	elif method == 'skf':
		raise NotImplementedError("This method is to be implemented later.")
	
def numpy_fill(arr: np.ndarray) -> np.ndarray:
	""" fills NaN values in a NumPy array with the last non-NaN value above it in the same column. """
	mask = np.isnan(arr)
	idx = np.where(~mask, np.arange(arr.shape[0]), 0)
	np.maximum.accumulate(idx, axis=0, out=idx)
	return arr[idx]
