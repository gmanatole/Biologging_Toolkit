import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import numpy as np
import calendar
import time

def get_start_time_sens(x) :
	return datetime.strptime(x, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()

def get_start_time_xml(path) :
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
