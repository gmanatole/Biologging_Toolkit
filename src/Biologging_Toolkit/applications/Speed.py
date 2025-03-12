import numpy as np
from scipy.optimize import curve_fit, Bounds
from Biologging_Toolkit.wrapper import Wrapper
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from glob import glob
import os

class Speed(Wrapper) :

	def __init__(self, 
			  depid, 
			  *, 
			  path, 
			  freq_min = 60,
			  freq_max = 80
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
		self.freq_min = freq_min
		self.freq_max = freq_max

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
				
		
	def from_inertial(self):
		"""
		Method to compute speed based on vertical speed and elevation angle.
		Moments where movement is in the horizontal plane are removed.
		"""
		angle = abs(self.ds['elevation_angle'][:].data)
		angle[angle < 0.35] = np.nan

		dP = abs(self.ds['depth'][:][1:] - self.ds['depth'][:].data[:-1])
		dP[dP < 0.5 / self.ds.sampling_rate] = np.nan
		
		speed = dP / np.sin(angle[:-1]) / self.ds.sampling_rate
		self.inertial_speed = np.append(speed, speed[-1])
		
	def fetch_acoustic(self, acoustic_path, overwrite = False):
		fns = np.sort(glob(os.path.join(acoustic_path, '*')))
		freq_min_idx = np.argmin(abs(np.load(fns[0])['freq'] - self.freq_min))
		freq_max_idx = np.argmin(abs(np.load(fns[0])['freq'] - self.freq_max))
		acoustic = []
		for fn in fns:
			acoustic.extend(np.nanmedian(np.load(fn)['spectro'][:, freq_min_idx:freq_max_idx+1], axis=1))
		metadata = {'long_name':'Acoustic data for speed estimation', 'units':'dB re 1uPa', 'freq_min':self.freq_min, 'freq_max':self.freq_max}
		self.create_variable('AFS', acoustic, self.ds['time'][:], overwrite=overwrite, **metadata)

	def from_acoustic(self):

		noise_level = self.ds['AFS'][:].data
		
		if 'inertial_speed' not in dir(self) :
			self.from_inertial()
			
		def func(x, a, b, c): 
			return a*x**2+b*x+c
			
		#classes = pd.cut(self.inertial_speed, bins=5, labels = [1,2,3,4,5]).to_numpy()
		#mask_nan = (~np.isnan(classes)) & (~np.isnan(noise_level))
		mask_nan = (~np.isnan(self.inertial_speed) & ~np.isnan(noise_level))
		acoustic_speed = np.full(len(noise_level), np.nan)

		mask_angle = self.ds['elevation_angle'][:].data <= -0.2
		mask = mask_angle & mask_nan
		popt, popv = curve_fit(func, noise_level[mask], self.inertial_speed[mask], maxfev=25000)
		acoustic_speed[mask_angle] = func(noise_level[mask_angle], *popt)

		mask_angle = abs(self.ds['elevation_angle'][:].data) < 0.2
		mask = mask_angle & mask_nan
		popt, popv = curve_fit(func, noise_level[mask], self.inertial_speed[mask], maxfev=25000)
		acoustic_speed[mask_angle] = func(noise_level[mask_angle], *popt)

		mask_angle = self.ds['elevation_angle'][:].data >= 0.2
		mask = mask_angle & mask_nan
		popt, popv = curve_fit(func, noise_level[mask], self.inertial_speed[mask], maxfev=25000)
		acoustic_speed[mask_angle] = func(noise_level[mask_angle], *popt)
		self.acoustic_speed = acoustic_speed

		"""skf = StratifiedKFold(n_splits=5)
		neg_params = []
		for i, (train_index, test_index) in enumerate(skf.split(noise_level[mask], classes[mask])):
			popt, popv = curve_fit(func, noise_level[mask][train_index], self.inertial_speed[mask][train_index], maxfev = 25000)
			acoustic_speed[mask][test_index] = func(noise_level[mask][test_index], *popt)
			neg_params.append(popt)
		self.neg_params = np.mean(neg_params, axis = 0)

		mask_angle = abs(self.ds['elevation_angle'][:].data) < 0.2
		mask = mask_angle & mask_nan
		skf = StratifiedKFold(n_splits=5)
		pos_params = []
		for i, (train_index, test_index) in enumerate(skf.split(noise_level[mask], classes[mask])):
			popt, popv = curve_fit(func, noise_level[mask][train_index], self.inertial_speed[mask][train_index], maxfev = 25000)
			acoustic_speed[mask][test_index] = func(noise_level[mask][test_index], *popt)
			pos_params.append(popt)
		self.pos_params = np.mean(pos_params, axis = 0)

		mask_angle = self.ds['elevation_angle'][:].data >= 0.2
		mask = mask_angle & mask_nan
		skf = StratifiedKFold(n_splits=5)
		pos_params = []
		for i, (train_index, test_index) in enumerate(skf.split(noise_level[mask], classes[mask])):
			popt, popv = curve_fit(func, noise_level[mask][train_index], self.inertial_speed[mask][train_index], maxfev = 25000)
			acoustic_speed[mask][test_index] = func(noise_level[mask][test_index], *popt)
			pos_params.append(popt)
		self.pos_params = np.mean(pos_params, axis = 0)"""

