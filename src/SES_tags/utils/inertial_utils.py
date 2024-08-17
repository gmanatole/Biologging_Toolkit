get_matlab_date = lambda x : calendar.timegm((datetime.datetime.fromordinal(int(x)) + datetime.timedelta(days=x%1) - datetime.timedelta(days = 366)).timetuple())
get_start_date = lambda x : calendar.timegm(time.strptime(x, '%Y/%m/%d %H:%M:%S'))


def angular_average(angles):
    # Convert angles to Cartesian coordinates
    x_coords = [np.cos(angle) for angle in angles]
    y_coords = [np.sin(angle) for angle in angles]
    
    # Compute the mean of the coordinates
    x_mean = np.nansum(x_coords) / len(angles)
    y_mean = np.nansum(y_coords) / len(angles)
    
    # Convert the mean coordinates back to an angle
    average_angle = np.arctan2(y_mean, x_mean)
    
    # Normalize the angle to be within [0, 2*pi)
    if average_angle < 0:
        average_angle += 2 * np.pi
    
    return average_angle

def modulo_pi(angle):
    # Normalize to range [0, 2*pi)
    angle = angle % (2 * np.pi)
    
    # Shift to range [-pi, pi)
    angle[angle > np.pi] = angle[angle > np.pi] - 2 * np.pi
    
    return angle


def coa(lat, lon):
	return np.sin(lat[1])*np.sin(lat[0]) + np.cos(lat[0])*np.cos(lat[1])*(np.cos((lon[1]-lon[0])))

def get_declination(lat, lon, time): 
	# NOAA API endpoint for declination data
	api_key = 'zNEw7'  # Replace with your actual API key

	# Additional parameters
	model = 'IGRF'
	time = datetime.datetime.fromtimestamp(time)
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

