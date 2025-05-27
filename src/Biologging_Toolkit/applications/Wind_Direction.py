import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
from Biologging_Toolkit.wrapper import Wrapper
from Biologging_Toolkit.utils.inertial_utils import angular_correlation, modulo_pi, angular_average
import pandas as pd



class WindDirection(Wrapper) :

	def __init__(self, 
			  depid, 
			  *, 
			  path, 
			  ):
		
		"""
		This class uses processed dataset to estimate the animal's speed.
		The main method is to use Euler angles to get the speed from the pitch and vertical speed.
		If acoustic data is available in the data structure a model can be fitted using the previous speed estimation.
		"""
		
		super().__init__(
			depid,
			path
        )


	def __call__(self, acoustic = True, overwrite = False) :
		self.forward(acoustic, overwrite)
		
	def forward(self, acoustic = True, overwrite = False):
		
		self.from_inertial()
		if overwrite :
			if 'inertial_speed' in self.ds.variables:
				self.remove_variable('inertial_speed')

		if 'inertial_speed' not in self.ds.variables:
			inertial = self.ds.createVariable('inertial_speed', np.float64, ('time',))
			inertial.units = 'm/s'
			inertial.long_name = 'Speed of the animal'
			inertial.measure = 'Computed using the elevation angle and the vertical speed'
			inertial.note = 'Dive portions where elephant seal has elevation angle below 0.35 rad or vertical speed below 0.5 m/s are removed'
			inertial[:] = self.inertial_speed
		
		if acoustic :
			self.from_acoustic()
			if overwrite :
				if 'acoustic_speed' in self.ds.variables:
					self.remove_variable('acoustic_speed')
	
			if 'acoustic_speed' not in self.ds.variables:
				acoustic = self.ds.createVariable('acoustic_speed', np.float64, ('time',))
				acoustic.units = 'm/s'
				acoustic.long_name = 'Speed of the animal'
				acoustic.measure = f'Computed using flow noise at {self.freq} and fitted on inertial speed'
				acoustic[:] = self.acoustic_speed
				
	def get_average_posture(self, t_off = 300):
		#Get azimuth as a function of elevation angle
		self.orientation = self.ds['azimuth'][:].data
		self.orientation[self.ds['elevation_angle'][:].data > 1.40] = self.ds['vertical_azimuth'][:].data[self.ds['elevation_angle'][:].data > 1.40]

		#Get surface times with dives longer than t_off
		ref_time = self.ds['time'][:].data
		time_data = ref_time[self.ds['depth'][:].data < 1]
		pos_dt = time_data[1:]-time_data[:-1]
		dives = np.where((pos_dt > t_off))[0]
		self.upper_bound, self.lower_bound = time_data[dives][1:], time_data[dives+1][:-1]
		avg_rot, avg_time, _rot, _time, len_rot = [], [], [], [], []

		#Get orientation data for those dives (remove beginning and end of surfacing)
		for time_up, time_down in zip(self.upper_bound, self.lower_bound) :
			len_rot.append(abs(time_up - time_down))
			rot_data = modulo_pi(self.orientation[(ref_time < time_up) & (ref_time > time_down)] + np.pi/2)
			_rot.append(angular_average(rot_data[5:-5]))
			_time.append(np.nanmean([time_up, time_down]))
		#Rolling average of surface orientations
		self.orientation_raw = _rot
		self.surface_time = _time
		self.orientation_conv = pd.Series(_rot).rolling(window = 15, min_periods = 5, center = True).median().to_numpy()
	    
	def get_correlation(self) :
		wind_orientation = interp1d(self.ds['time'][:].data, np.arctan2(self.ds['v10'][:].data, self.ds['u10'][:].data), bounds_error = False)(self.surface_time)
		#y1 = interp(aux.time, modulo_pi(aux.mdts*np.pi/180+np.pi/2), bounds_error = False)(_time)
		#y2 = interp(aux.time, modulo_pi(aux.mdww*np.pi/180+np.pi/2), bounds_error = False)(_time)
		#y3 = interp(aux.time, modulo_pi(aux.mwd*np.pi/180+np.pi/2), bounds_error = False)(_time)
		#y4 = interp(aux.time, modulo_pi(aux.dwi*np.pi/180+np.pi/2), bounds_error = False)(_time)
		corr = angular_correlation(self.orientation_conv, wind_orientation)
		self.positive_corr = corr[0]
		self.negative_corr = corr[1]


'''if plot_errors :
	cmap = cm.get_cmap('Reds')
	az = modulo_pi(interp(aux.time, aux.sun_azimuth, bounds_error = False)(_time))
	zen = interp(aux.time, aux.sun_zenith, bounds_error = False)(_time)
	wind_dir = modulo_pi(interp(aux.time, np.arctan2(aux.v10, aux.u10), bounds_error=False)(_time))
	y1 = interp(aux.time, np.sqrt(aux.u10**2 + aux.v10**2), bounds_error = False)(_time)
	y2 = interp(aux.time, modulo_pi(aux.mdww*np.pi/180+np.pi/2), bounds_error = False)(_time)
	y3 = interp(aux.time, modulo_pi(aux.mwd*np.pi/180+np.pi/2), bounds_error = False)(_time)
	y4 = interp(aux.time, modulo_pi(aux.dwi*np.pi/180+np.pi/2), bounds_error = False)(_time)

	azimuth = abs(modulo_pi(az+np.pi-wind_dir))
	
	err = abs(modulo_pi(wind_dir - orientation_conv))
	fig, ax = plt.subplots(1,3, figsize = (15, 7))
	ax = ax.flatten()
	ax[0].scatter(azimuth, err)
	ax[1].scatter(zen, err)
	ax[2].scatter(y1, err)

plt.show()'''
    
