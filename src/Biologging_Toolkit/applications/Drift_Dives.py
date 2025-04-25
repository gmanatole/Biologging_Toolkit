import numpy as np
from glob import glob
import os
import umap.umap_ as umap
import hdbscan
from tqdm import tqdm
from Biologging_Toolkit.wrapper import Wrapper
from Biologging_Toolkit.utils.bioluminescence_utils import find_sequence

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


	def from_acceleration(self, acc_threshold = 0.2, dur_threshold = 200, depth_threshold = None):
		"""
		Code translated from Matlab code used by Richard et al., 2014 to find drift dives using acceleration
		Parameters
		----------
		acc_threshold : Acceleration threshold above which behavior is considered as active swimming. Default is 0.2 m/s2.
		dur_threshold :  Minimum duration (in seconds) of drift dives to keep. Default is 200s.
		depth_threshold : Maximum depth under which drift dives are not considered. Default is None.
		Returns
		-------
		Creates three attributes by adding each threshold one by one.
		"""
		res = 1
		X, Y, Z = self.sens['A'][:].data
		samplerate = self.sens['A'].sampling_rate
		# Compute acceleration norm
		SOM = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
		# Compute STD of acceleration over sliding window
		StdFix = []
		for ii in range(0, len(SOM) - int(res * samplerate), int(res * samplerate)):
			StdFix.append(np.nanstd(SOM[ii:ii + int(res * samplerate)]))
		StdFix = np.array(StdFix)
		# Make vector same dimension as sens5
		if len(StdFix) > len(SOM):
			StdFix = StdFix[:len(SOM)]
		else:
			padded = np.full(len(SOM), np.nan)
			padded[:len(StdFix)] = StdFix
			StdFix = padded
		# STD on 40s sliding window
		VarMob = np.full(len(StdFix), np.nan)
		for ii in range(int(20 * samplerate), len(StdFix) - int(20 * samplerate)):
			VarMob[ii] = np.nanstd(
				StdFix[ii - int(20 * samplerate): ii + int(20 * samplerate)])
		# Find length and timestamps of periods respecting this condition
		tab = find_sequence(VarMob < acc_threshold, samplerate=samplerate)
		I = np.where((tab[:, 3] > dur_threshold) & (tab[:, 0] == 1))[0]
		# Keep periods where depth is higher than threshold
		J = []
		if depth_threshold :
			for ii in I:
				segment = self.sens['P'][:].data[int(tab[ii, 1]): int(tab[ii, 2]) + 1]
				T = np.where(segment >= depth_threshold)[0]
				if len(T) * samplerate > dur_threshold:
					T_diff = np.diff(T)
					verif = find_sequence(T_diff == 1, samplerate = samplerate)
					if np.any((verif[:, 0] == 1) & (verif[:, 3] >= 30)):
						J.append(int(tab[ii, 1]))
		self.drifts = tab
		self.long_drifts = I
		self.upper_drifts = J

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


	def acoustic_cluster(self, acoustic_path = None, min_cluster_size = 50, min_samples = 10):
		fns = glob(os.path.join(acoustic_path, '*'))
		nfeatures = 15
		posfeatures = np.exp(np.arange(0, nfeatures) * np.log(513) / nfeatures).astype(int)
		X = []
		_fns = []
		for fn in fns:
			data = np.load(fn)
			if data['len_spectro'] <= 300:
				continue
			_fns.append(fn)
			X.append(np.load(fn)['spectro'][:300:30, posfeatures])
		self.cluster_fns = np.array(_fns)
		X = np.array(X)
		X = X.reshape(X.shape[0], -1)
		project = umap.UMAP()
		embed = project.fit_transform(X)
		self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(embed)

	def acoustic_threshold(self, threshold = None):
		if not threshold :
			self.from_acceleration()
		pass



	
