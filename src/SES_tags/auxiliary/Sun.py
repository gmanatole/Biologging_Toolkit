def azimuth_to_trigonometric(azimuth):
    trigonometric_angle = 450 - azimuth
    trigonometric_angle = trigonometric_angle % 360
    return trigonometric_angle

def get_sun_pos(aux):
	loc_ses = coord.EarthLocation(lat = aux.lat * u.deg, lon = aux.lon * u.deg, height = 0 * u.m)
	time_ses = Time(aux.time, format='unix', scale = 'utc')
	sun = coord.get_sun(time_ses)
	altaz = coord.AltAz(location=loc_ses, obstime=time_ses)
	return np.array(sun.transform_to(altaz).az)*np.pi/180, np.array(sun.transform_to(altaz).zen)*np.pi/180
	
