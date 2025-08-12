import numpy as np
from astropy.time import Time
import astropy.units as u
import astropy.coordinates as coord

def azimuth_to_trigonometric(azimuth):
    trigonometric_angle = 450 - azimuth
    trigonometric_angle = trigonometric_angle % 360
    return trigonometric_angle

def get_sun_pos(timestamp, lat, lon):
	loc_ses = coord.EarthLocation(lat = lat * u.deg, lon = lon * u.deg, height = 0 * u.m)
	timestamp = Time(timestamp, format='unix', scale = 'utc')
	sun = coord.get_sun(timestamp)
	altaz = coord.AltAz(location=loc_ses, obstime=timestamp)
	return np.array(sun.transform_to(altaz).az)*np.pi/180, np.array(sun.transform_to(altaz).zen)*np.pi/180, np.array(sun.transform_to(altaz).alt*np.pi/180)
	
