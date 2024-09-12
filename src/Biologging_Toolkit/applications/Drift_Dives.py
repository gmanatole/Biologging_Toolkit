import matplotlib.pyplot as plt
import matplotlib.colors as co
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import umap.umap_ as umap
import hdbscan
import scipy
from tqdm import tqdm
from Biologging_Toolkit.wrapper import Wrapper

class DriftDives(Wrapper) :
	"""
	A class for analyzing drift dives in animals.
	
	Attributes:
		depid: The deployment ID of the animal being analyzed.
		analysis_length: The length of the dive portion (in seconds) that is analyzed to be considered as a drift dive.
	"""
	
	def __init__(self, 
			  depid, 
			  *, 
			  path,
			  analysis_length = 60
			  ):
		"""
		Initializes the DriftDives object with the given deployment ID, dataset path, and analysis length.
		
		Args:
			depid: The deployment ID of the animal being analyzed.
			path: The path to the dataset, must contain either inertial or depth data.
			analysis_length: The length in seconds of the dive portion to be analyzed as a drift dive. Default is 60 seconds.
		"""

		super().__init__(
			depid,
			path
			)
		self.analysis_length = analysis_length

	def __call__(self, overwrite = False, mode = 'inertial') :
		self.forward(mode, overwrite)
		
	def forward(self, mode = 'inertial', overwrite = False):
		"""
		Calls the  correct method for drift dive identification in the specified mode.
		Updating the dataset with drift information.
		
		Args:
			mode: The mode of analysis. Can be 'inertial' (uses bank angle) or 'depth' (uses vertical speed). Default is 'inertial'.
			overwrite: If True, existing drift variables in the dataset will be removed before analysis. Default is False.
		Mode-specific behavior:
			- 'inertial': Drift portions are flagged where the animal has a bank angle greater than 2 radians for 98% of the analysis length.
			- 'depth': Drift portions are flagged where the animal's vertical speed is below 0.4 m/s for the analysis length.
		"""
		if mode == 'inertial' :
			self.from_inertial()
			if overwrite :
				if 'inertial_drift' in self.ds.variables:
					self.remove_variable('inertial_drift')
	
			if 'inertial_speed' not in self.ds.variables:
				inertial = self.ds.createVariable('inertial_drift', np.float64, ('time',))
				inertial.units = 'binary'
				inertial.long_name = 'Flag indicating whether or not the animal is passively drifting'
				inertial.measure = '1 for drift portions, 0 for any other behavior'
				inertial.analysis_length = self.analysis_length
				inertial.note = 'Dive portions where elephant seal has bank angle greater than 2 rad for 98% of analysis length'
				inertial[:] = self.inertial_drift		
				
		if mode == 'depth' :
			self.from_inertial()
			if overwrite :
				if 'depth_drift' in self.ds.variables:
					self.remove_variable('depth_drift')
	
			if 'depth_speed' not in self.ds.variables:
				depth = self.ds.createVariable('depth_drift', np.float64, ('time',))
				depth.units = 'binary'
				depth.long_name = 'Flag indicating whether or not the animal is passively drifting'
				depth.measure = '1 for drift portions, 0 for any other behavior'
				depth.analysis_length = self.analysis_length
				depth.note = 'Dive portions where elephant seal has average vertical speed below 0.4 m/s during analysis length'
				depth[:] = self.depth_drift


	def from_inertial(self):
		"""
		Identifies drift portions based on the animal's bank angle ('inertial' mode).
		
		Analyzes sections of the dive where the bank angle exceeds 2 radians for more than 98% of the analysis length.
		The results are stored in `inertial_drift`, a binary flag in the dataset.
		"""
		idx_bound = int(self.analysis_length / self.ds.sampling_rate / 2)
		dive_type = np.full(len(self.ds['bank_angle'][:]), 0)
		dive_type = np.full(len(self.ds['depth'][:]), 0)
		pbar = tqdm(total = len(dive_type)-2*idx_bound-1, position = 0, leave = True)
		pbar.set_description('Iterating through dataset')
		for j in range(idx_bound, len(dive_type)-idx_bound-1) :
			if ((abs(self.ds['bank_angle'][:][j-self.analysis_length:j+self.analysis_length]) > 2).sum() / self.analysis_length > 0.98):
				dive_type[j] =  1 
			pbar.update(1)
		self.inertial_drift = dive_type

	def from_depth(self):
		"""
		Identifies drift portions based on the animal's vertical speed ('depth' mode).
		
		Analyzes sections of the dive where the average vertical speed is below 0.4 m/s over the analysis length.
		The results are stored in `inertial_drift`, a binary flag in the dataset.
		"""
		idx_bound = int(self.analysis_length / self.sampling_rate)
		dive_type = np.full(len(self.ds['depth'][:]), 0)
		pbar = tqdm(total = len(dive_type)-2*idx_bound-1, position = 0, leave = True)
		pbar.set_description('Iterating through dataset')
		for j in range(idx_bound, len(dive_type)-idx_bound-1) :
			vertical_speed_before = (self.ds['depth'][:][j] - self.ds['depth'][:][j-idx_bound])/(self.ds['time'][:][j] - self.ds['time'][:][j-idx_bound]) 
			vertical_speed_after = (self.ds['depth'][:][j+idx_bound] - self.ds['depth'][:][j])/(self.ds['time'][:][j+idx_bound] - self.ds['time'][:][j]) 
			if (abs(vertical_speed_after) < 0.4) or abs((vertical_speed_before) < 0.4) :
				dive_type[j] = 1
			pbar.update(1)
		self.depth_drift = dive_type


	def acoustic_cluster(self, ml, var = '2200', min_cluster_size = 20, min_samples = 15):
		spl_df = pd.read_csv(f'/run/media/grosmaan/LaCie/individus_brut/individus/{ml}/spl_data_cal.csv').drop('time', axis = 1)
		aux_df = pd.read_csv(f'/run/media/grosmaan/LaCie/individus_brut/individus/{ml}/aux_data.csv')
		df = pd.concat((aux_df, spl_df), axis = 1).dropna(subset = var)
		indices = get_indices(df.depth)
		
		X = np.zeros((len(indices)-1, 100))
		data = pd.DataFrame()
		timestamp, depth = np.zeros((len(indices)-1, 100)), np.zeros((len(indices)-1, 100))
		for i in range(1, len(indices)):
			X[i-1] = scipy.signal.resample(df[var].iloc[indices[i-1]:indices[i]], 100)
			timestamp[i-1] = np.linspace(df.time.iloc[indices[i-1]], df.time.iloc[indices[i]], 100)
			depth[i-1] = scipy.signal.resample(df.depth.iloc[indices[i-1]:indices[i]], 100)
		#X = np.lib.stride_tricks.sliding_window_view(df[var], window_shape=(window_size))
		#data = pd.DataFrame()
		#data['depth'] = df.depth[window_size//2 : -window_size//2+1]
		#data['timestamp'] = df.time[window_size//2 : -window_size//2+1]


		project = umap.UMAP()
		embed = project.fit_transform(X)
		
		clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(embed) 
		data['cluster'] = clusterer.labels_
		data.cluster[data.cluster == -1] = -100
		norm = co.Normalize(vmin=clusterer.labels_.min(), vmax=clusterer.labels_.max())
		cmap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
		for i in range(len(indices)-1):
			plt.scatter(timestamp[i], depth[i], color = cmap.to_rgba([int(clusterer.labels_[i])]), s = 2)
		#plt.plot(data.timestamp, data.depth, c = data.cluster)
		#plt.plot(df.time, df[var])
		plt.show()
		return timestamp, depth, clusterer.labels_, embed

	def acoustic_threshold(self):
		pass



	
