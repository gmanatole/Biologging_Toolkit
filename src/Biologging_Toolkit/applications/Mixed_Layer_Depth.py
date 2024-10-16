import numpy as np
from Biologging_Toolkit.processing.Dives import Dives
from Biologging_Toolkit.utils.format_utils import *
from tqdm import tqdm 
import pandas as pd

class MixedLayerDepth(Dives) :
	
	def __init__(self,
			  depid : str,
			  *,
			  path : str = None  
			  ):
		
				
		super().__init__(
			depid,
			path = path
        )
		
		self.depth = self.ds['depth'][:].data
		self.temp = self.ds['temperature'][:].data
		self.dives = self.ds['dives'][:].data
		self.criterion = 0.03
		
	def __call__(self, overwrite = False):
		return self.forward(overwrite = overwrite)
	
	def forward(self, window_size = 10, overwrite = True):
		self.compute_mld()
		zn1_corr, zn2_corr = [], []
		for i in range(len(self.zn1)):
		    if self.zn1[i] != 0:
		        zn1_corr.append(np.nanmean(self.zn1[i]))
		    else :
		        zn1_corr.append(np.nan)
		    if self.zn2[i] != 0:
		        zn2_corr.append(np.nanmean(self.zn2[i]))
		    else :
		        zn2_corr.append(np.nan)
		zn1_series = pd.Series(zn1_corr)
		self.zn1_smoothed = zn1_series.rolling(window=window_size, min_periods=1).mean().to_numpy()
		zn2_series = pd.Series(zn2_corr)
		self.zn2_smoothed = zn2_series.rolling(window=window_size, min_periods=1).mean().to_numpy()
		self.dive_ds['zn1'] = self.zn1_smoothed
		self.dive_ds['zn2'] = self.zn2_smoothed
		self.dive_ds.to_csv(self.dive_path, index = None)
	
	
	@staticmethod
	def profile_check(depth, profile, sr) :
		if len(profile) >= 3 * 60 / sr :
			indices = np.argsort(depth)
			depth = depth[indices]
			profile = profile[indices]
			return depth[depth > 5], profile[depth > 5]
		else :
			return [], []
		
	def compute_mld(self):
		up, down = self.get_dive_direction(self.depth[:: 60 // self.ds.sampling_rate])
		up, down = resample_boolean_array(up, len(self.depth)), resample_boolean_array(down, len(self.depth))
		self.zn2, self.zn1 =[], []
		unique_dives = np.unique(self.dives)
		#Look at each dive individually
		for dive in tqdm(unique_dives, desc = 'Computing MLD for each dive') :
			_zn1, _zn2 = [], []
			# Get estimations from two profiles : up and down trajectories
			for direction in [up, down]:
				_temp = self.temp[(self.dives == dive) & direction]
				_depth = self.depth[(self.dives == dive) & direction]
				#Sort temp and depth profile by increasing depth and check if there is enough data
				_depth, _temp = self.profile_check(_depth, _temp, self.ds.sampling_rate)
				delta_k, std_k, X_k = [],[],[]
				for k in range(len(_temp)):
					delta_k.append(np.sqrt( 1/(k+1) * np.nansum([((_temp[i] - np.nanmean(_temp[:k+1]))**2) for i in range(k)])))
					std_k.append(np.nanmax(_temp[:k+1]) - np.nanmin(_temp[:k+1]))
					X_k.append(delta_k[-1] / std_k[-1])
				#If X_k contains only nan values, add nan
				try : 
					pos_zn1 = np.nanmin((np.nanargmin(X_k), len(X_k)-2))  #If lowest X_k is last element, take penultimate element
					_zn1.append(_depth[pos_zn1])
				except ValueError :
					_zn1.append(np.nan)
					_zn2.append(np.nan)
					continue
				for step in range(len(delta_k[:pos_zn1 - 1]), -1, -1):
					if (delta_k[step + 1] - delta_k[step]) / (delta_k[pos_zn1 + 1] - delta_k[pos_zn1]) <= self.criterion :
						_zn2.append(_depth[step])
						break
			self.zn2.append(_zn2)
			self.zn1.append(_zn1)
		

import matplotlib.pyplot as plt
def plot(inst, aux, window_size = 10):
	mld_corr = []
	for i in range(len(inst.mld)):
	    if inst.mld_estimation[i] != 0:
	        mld_corr.append(np.nanmean(inst.mld_estimation[i]))
	    else :
	        mld_corr.append(np.nan)		
	signal_series = pd.Series(mld_corr)
	smoothed_signal = signal_series.rolling(window=window_size, min_periods=1).mean().to_numpy()
	time_ds = inst.ds['time'][:].data
	time_dives = []
	for i in np.unique(inst.dives):
	    time_dives.append(np.mean(time_ds[inst.dives == i]))
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(time_dives, smoothed_signal, '-o', label='MLD Estimation')
	ax2.plot(aux.time, aux.era, label='ERA5 Wind Speed', color='orange')
	lines_1, labels_1 = ax1.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
	ax1.set_ylabel('MLD Depth (m)')
	ax2.set_ylabel('Wind Speed (m/s)')
	plt.grid()
	plt.show()