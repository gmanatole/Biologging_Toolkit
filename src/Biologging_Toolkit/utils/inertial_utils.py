from datetime import datetime
import numpy as np
import requests
import xml.etree.ElementTree as ET

def find_sequence(condition, samplerate =  1):
    condition = condition.astype(int)
    edges = np.diff(condition, prepend=0, append=0)
    start_idx = np.where(edges == 1)[0]
    end_idx = np.where(edges == -1)[0] - 1
    length = (end_idx - start_idx + 1) / samplerate
    result = np.vstack([condition[start_idx], start_idx, end_idx, length]).T
    return result

def norm(X):
	m, n = X.shape
	if m == 1 or n == 1:
	    v = np.linalg.norm(X)
	else:
	    v = np.sqrt(np.abs(X)**2 @ np.ones((n, 1)))
	return v

def angular_average(angles):
    x_coords = [np.cos(angle) for angle in angles]
    y_coords = [np.sin(angle) for angle in angles]
    x_mean = np.nansum(x_coords) / len(angles)
    y_mean = np.nansum(y_coords) / len(angles)
    average_angle = np.arctan2(y_mean, x_mean)
    if average_angle < 0:
        average_angle += 2 * np.pi
    return average_angle

def modulo_pi(angle):
    angle = angle % (2 * np.pi)
    angle[angle > np.pi] = angle[angle > np.pi] - 2 * np.pi
    return angle

def angular_correlation(x,y):
    dpos = x - y
    dneg = x + y
    pos_corr = np.sqrt((np.nansum(np.cos(dpos))**2 + np.nansum(np.sin(dpos))**2))/len(dpos)
    neg_corr = np.sqrt((np.nansum(np.cos(dneg))**2 + np.nansum(np.sin(dneg))**2))/len(dneg)
    return pos_corr, neg_corr

def coa(lat, lon):
	return np.sin(lat[1])*np.sin(lat[0]) + np.cos(lat[0])*np.cos(lat[1])*(np.cos((lon[1]-lon[0])))

def get_declination(lat, lon, time): 
	# NOAA API endpoint for declination data
	api_key = 'zNEw7'  # Replace with your actual API key

	# Additional parameters
	model = 'IGRF'
	time = datetime.fromtimestamp(time)
	start_year = time.year
	start_month = time.month
	start_day = time.day
	result_format = 'xml'  # You can change this to 'json', 'csv', or 'html' as needed

	url = f'https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination?lat1={lat}&lon1={lon}&key={api_key}&model={model}&startYear={start_year}&startMonth={start_month}&startDay={start_day}&resultFormat={result_format}'
	response = requests.get(url)
	if response.status_code == 200:
		declination_data = response.text
		root = ET.fromstring(declination_data)
		for elem in root.iter():
			  if elem.tag == 'declination':
			      declination = float(elem.text)
		return declination
	else:
		return 0

