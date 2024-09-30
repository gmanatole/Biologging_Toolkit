import xml.etree.ElementTree as ET
from datetime import datetime, timezone
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
	Enter path and dictionary containing keys (must be column name from d3sensordefs file) and values for thoses keys
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
	return [np.where(chans_array == num)[0] for num in num_name]
	
	
	
def get_boundaries_metadata(path) :
	"""
	Enter path to built data structure to get time and space boundaries.
	This data can then be used to download ERA
	"""
	ds = nc.Dataset(path)
	time_min = datetime.fromtimestamp(np.nanmin(ds['time'][:])).replace(minute = 0, second = 0)
	time_max = datetime.fromtimestamp(np.nanmax(ds['time'][:]))
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
	
	
	
	
	
	
	
	
	
	
	
