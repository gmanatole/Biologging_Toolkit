import numpy as np
from scipy.optimize import curve_fit, Bounds
from Biologging_Toolkit.wrapper import Wrapper
from sklearn.model_selection import StratifiedKFold
import pandas as pd


class Speed(Wrapper) :

	def __init__(self, 
			  depid, 
			  *, 
			  path, 
			  threshold = 20
			  ):
		
		"""
		This class uses processed dataset to estimate the animal's speed.
		The main method is to use Euler angles to get the speed from the pitch and vertical speed.
		If acoustic data is available in the data structure a model can be fitted using the previous speed estimation.
		"""
		
		self.threshold = threshold
		super().__init__(
			depid,
			path
        )

	def __call__(self, overwrite = False) :
		self.forward()
		
	def forward(self, overwrite = False, acoustic = True):
		
		self.from_inertial()
		if acoustic :
			self.from_acoustic()
		
		
	def from_inertial(self):
		"""
		Method to compute speed based on vertical speed and elevation angle.
		Moments where movement is in the horizontal plane are removed.
		"""
		angle = abs(self.ds['elevation_angle'][:].data)
		angle[angle < 0.35] = np.nan

		dP = abs(self.ds['depth'][:][1:] - self.ds['depth'][:].data[:-1])
		dP[dP < 0.5] = np.nan
		
		speed = dP / np.sin(angle[:-1]) / self.ds.sampling_rate
		self.inertial_speed = np.append(speed, speed[-1])
		

	def from_acoustic(self, freq = 60):
		
		freq_idx = np.argmin(abs(self.ds['frequency_spectrogram'][:] - freq))
		noise_level = self.ds['spectrogram'][:, freq_idx].data
		
		if 'inertial_speed' not in dir(self) :
			self.from_inertial()
			
		def func(x, a, b, c): 
			return a*x**2+b*x+c
			
		depth = [self.threshold,np.inf]  #depth (m) threshold
		classes = pd.cut(self.inertial_speed, bins=5, labels = [1,2,3,4,5]).to_numpy()
		
		mask_nan = (~np.isnan(classes)) & (~np.isnan(noise_level))
		
		acoustic_speed = np.full(len(noise_level), np.nan)

		mask_angle = self.ds['elevation_angle'][:].data < 0
		mask = mask_angle & mask_nan
		skf = StratifiedKFold(n_splits=5)
		neg_params = []
		for i, (train_index, test_index) in enumerate(skf.split(noise_level[mask], classes[mask])):
			popt, popv = curve_fit(func, noise_level[mask][train_index], self.inertial_speed[mask][train_index], maxfev = 25000)
			estimation = func(noise_level[mask][test_index], *popt)
			neg_params.append(popt)
		self.neg_params = np.mean(neg_params, axis = 0)
		acoustic_speed[mask] = func(noise_level[mask], *self.neg_params)

				  
		mask_angle = self.ds['elevation_angle'][:].data >= 0
		mask = mask_angle & mask_nan
		skf = StratifiedKFold(n_splits=5)
		pos_params = []
		for i, (train_index, test_index) in enumerate(skf.split(noise_level[mask], classes[mask])):
			popt, popv = curve_fit(func, noise_level[mask][train_index], self.inertial_speed[mask][train_index], maxfev = 25000)
			estimation = func(noise_level[mask][test_index], *popt)
			pos_params.append(popt)
		self.pos_params = np.mean(pos_params, axis = 0)
		acoustic_speed[mask] = func(noise_level[mask], *self.pos_params)

		self.acoustic_speed = acoustic_speed
