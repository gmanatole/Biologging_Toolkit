import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np

def get_start_time_xml(path) :
	if isinstance(path, (str, np.str_)):
		tree = ET.parse(path)
		root = tree.getroot()
		for cfg in root.findall('CFG'):
		    if cfg.find('RANGE[@SENS="ACC"]') != None:
		        return datetime.strptime(cfg.attrib.get('TIME'), '%Y,%m,%d,%H,%M,%S').timestamp()
		raise ValueError(f'starting time not found in xml file {path}')

	if isinstance(path, (list, np.ndarray)) :
		xml_dates = []
		for i, fn in enumerate(path) :
			try :
				tree = ET.parse(fn)
				root = tree.getroot()
				for cfg in root.findall('CFG'):
					if cfg.find('RANGE[@SENS="ACC"]') != None :
						xml_dates.append(datetime.strptime(cfg.attrib.get('TIME'), '%Y,%m,%d,%H,%M,%S').timestamp())
				if i+1 != len(xml_dates) :
					print(f'starting time not found in xml file {path[i]}')
			except ET.ParseError :
				continue
		return np.array(xml_dates)
	
	
