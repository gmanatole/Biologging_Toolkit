import numpy as np
from glob import glob
import os
import umap.umap_ as umap
import hdbscan
import netCDF4 as nc
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, f1_score
from typing import Union, List
from Biologging_Toolkit.wrapper import Wrapper
from Biologging_Toolkit.utils.inertial_utils import find_sequence
from Biologging_Toolkit.utils.acoustic_utils import sort_spectrogram
from Biologging_Toolkit.utils.format_utils import get_start_time_sens

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
			  sens_path = None,
			  ):
		"""
		Initializes the DriftDives object with the given deployment ID, dataset path, and analysis length.
		Args:
			depid: The deployment ID of the animal being analyzed.
			path: The path to the dataset, must contain either inertial or depth data.
		"""

		super().__init__(
			depid,
			path
			)
		self.sens_path = sens_path

	def __call__(self, overwrite = False, mode = 'inertial') :
		self.forward(mode, overwrite)
		
	def forward(self, mode = 'inertial', overwrite = False, acc_method = 'long'):
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
	
			if 'inertial_drift' not in self.ds.variables:
				inertial = self.ds.createVariable('inertial_drift', np.float64, ('time',))
				inertial.units = 'binary'
				inertial.long_name = 'Flag indicating whether or not the animal is passively drifting'
				inertial.measure = '1 for drift portions, 0 for any other behavior'
				inertial.analysis_length = self.analysis_length
				inertial.note = 'Dive portions where elephant seal has bank angle greater than 2 rad for 98% of analysis length'
				inertial[:] = self.inertial_drift		
				
		if mode == 'depth' :
			self.from_depth()
			if overwrite :
				if 'depth_drift' in self.ds.variables:
					self.remove_variable('depth_drift')
	
			if 'depth_drift' not in self.ds.variables:
				depth = self.ds.createVariable('depth_drift', np.float64, ('time',))
				depth.units = 'binary'
				depth.long_name = 'Flag indicating whether or not the animal is passively drifting'
				depth.measure = '1 for drift portions, 0 for any other behavior'
				depth.analysis_length = self.analysis_length
				depth.note = 'Dive portions where elephant seal has average vertical speed below 0.4 m/s during analysis length'
				depth[:] = self.depth_drift

		if mode == 'acceleration':
			self.from_acceleration()
			if acc_method == 'long':
				_acc = self.drifts[self.long_drifts]
			elif acc_method == 'upper' :
				_acc = self.drifts[self.upper_drifts]
			else :
				_acc = self.drifts
			timestamps = get_start_time_sens(self.sens.dephist_device_datetime_start) + np.arange(0, len(
				self.sens['A'][0]) - self.sens['A'].sampling_rate) / self.sens['A'].sampling_rate
			timestamps = timestamps[::int(self.sens['A'].sampling_rate)]
			drift = np.zeros((len(timestamps)))
			for _drift in _acc:
				drift[int(_drift[1]):int(_drift[2])] = 1
			drifts = interp1d(timestamps, drift, bounds_error = False)(self.ds['time'][:].data)
			if overwrite :
				if 'acc_drift' in self.ds.variables:
					self.remove_variable("acc_drift")
			if 'acc_drift' not in self.ds.variables:
				acc = self.ds.createVariable('acc_drift', np.float64, ('time',))
				acc.units = 'binary'
				acc.long_name = 'Flag indicating whether or not the animal is passively drifting'
				acc.measure = '1 for drift portions, 0 for any other behavior'
				acc.method = 'Threshold on acceleration standard deviation'
				acc.acc_threshold = self.acc_threshold
				acc.dur_threshold = self.dur_threshold
				acc.depth_threshold = self.depth_threshold
				acc.analysis_length = '1s'
				acc.note = f'Dive portions where elephant seal has average acceleration under {self.acc_threshold} during analysis length'
				acc[:] = drifts


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
		self.sens = nc.Dataset(self.sens_path)
		res = 1
		X, Y, Z = self.sens['A'][:].data
		samplerate = self.sens['A'].sampling_rate
		# Compute acceleration norm
		SOM = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
		# Compute STD of acceleration over sliding window
		StdFix = []
		pbar = tqdm(total = int((len(SOM)-int(res*samplerate))/(res*samplerate))+int(len(SOM)/(res*samplerate)-40*samplerate), position = 0, leave = True)
		pbar.set_description('Finding drift dives using acceleration')
		for ii in range(0, len(SOM) - int(res * samplerate), int(res * samplerate)):
			pbar.update(1)
			StdFix.append(np.nanstd(SOM[ii:ii + int(res * samplerate)]))
		StdFix = np.array(StdFix)
		# STD on 40s sliding window
		VarMob = np.full(len(StdFix), np.nan)
		for ii in range(int(20 * samplerate), len(StdFix) - int(20 * samplerate)):
			pbar.update(1)
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
		self.acc_threshold = acc_threshold
		self.depth_threshold = depth_threshold if depth_threshold is not None else 0
		self.dur_threshold = dur_threshold

	def from_inertial(self, analysis_length = 60):
		"""
		Identifies drift portions based on the animal's bank angle ('inertial' mode).
		
		Analyzes sections of the dive where the bank angle exceeds 2 radians for more than 98% of the analysis length.
		The results are stored in `inertial_drift`, a binary flag in the dataset.
		Parameters
		----------
		analysis_length :  Length (in s) of sliding window for vertical speed smoothing.
		"""
		idx_bound = int(analysis_length / self.ds.sampling_rate / 2)
		dive_type = np.full(len(self.ds['depth'][:]), 0)
		pbar = tqdm(total = len(dive_type)-2*idx_bound-1, position = 0, leave = True)
		pbar.set_description('Iterating through dataset')
		for j in range(idx_bound, len(dive_type)-idx_bound-1) :
			if ((abs(self.ds['bank_angle'][:][j-analysis_length:j+analysis_length]) > 2).sum() / analysis_length > 0.98):
				dive_type[j] =  1 
			pbar.update(1)
		self.inertial_drift = dive_type
		self.analysis_length = analysis_length

	def from_depth(self, speed_threshold = 0.6, smoothing_length = 10, drift_length = 180):
		"""
		Identifies drift portions based on the animal's vertical speed ('depth' mode).
		Based on Dragon et al., 2012
		Analyzes sections of the dive where the average vertical speed is below 0.4 m/s over the analysis length.
		The results are stored in `depth_drift`, a binary flag in the dataset.
		Parameters
		----------
		speed_threshold : Vertical speed threshold above which behavior is considered as active swimming. Default is 0.6 m/s.
		smoothing_length :  Length (in s) of sliding window for vertical speed smoothing.
		drift_length : Length (in s) of portions to consider as drift dives.
		"""
		# Compute vertical speed
		vertical_speed = np.full(len(self.ds['depth'][:]), np.nan)
		_vertical_speed = np.full(len(self.ds['depth'][:]), np.nan)
		_vspeed = np.diff(self.ds['depth'][:].data) / np.diff(self.ds['time'][:].data)
		_vertical_speed[:len(_vspeed)] = _vspeed

		#Apply smoothing window
		speed_bound = int(np.ceil(smoothing_length / 2 / self.ds.sampling_rate))
		pbar = tqdm(total = len(vertical_speed)-2*speed_bound-1, position = 0, leave = True)
		pbar.set_description('Applying smoothing window')
		for j in range(speed_bound, len(vertical_speed) - speed_bound - 1) :
			vertical_speed[j] = np.nanmean(_vertical_speed[j - speed_bound : j + speed_bound])
			pbar.update(1)

		# Find drift dives
		dive_type = np.full(len(self.ds['depth'][:]), 0)
		drift_bound = int(np.ceil(drift_length / self.ds.sampling_rate / 2))
		pbar = tqdm(total = len(dive_type)-2*drift_bound-1,position = 0, leave = True)
		pbar.set_description('Finding drift_dives')
		for k in range(drift_bound, len(dive_type)-drift_bound-1) :
			if (np.all(abs(vertical_speed[k - drift_bound : k + drift_bound]) <= speed_threshold)) & (np.nanstd(vertical_speed[k - drift_bound : k + drift_bound])**2 < 0.005) :
				dive_type[k - drift_bound : k + drift_bound] = 1
			pbar.update(1)
		self.depth_drift = dive_type
		self.analysis_length = drift_length
		#idx_bound = int(analysis_length / self.sampling_rate)
		#for j in range(idx_bound, len(dive_type)-idx_bound-1) :
			#vertical_speed_before = (self.ds['depth'][:][j] - self.ds['depth'][:][j-idx_bound])/(self.ds['time'][:][j] - self.ds['time'][:][j-idx_bound])
			#vertical_speed_after = (self.ds['depth'][:][j+idx_bound] - self.ds['depth'][:][j])/(self.ds['time'][:][j+idx_bound] - self.ds['time'][:][j])
			#if (abs(vertical_speed_after) < speed_threshold) or abs((vertical_speed_before) < speed_threshold) :
			#	dive_type[j] = 1

	def acoustic_cluster(self,
						 nfeatures = 15,
						 freqs = None,
						 timestep = 5,
						 tmin = 0,
						 tmax = 15,
						 bathy = [0, 20000],
						 acoustic_path = None,
						 min_cluster_size = 50,
						 min_samples = 10,
						 sort = False,
						 freq_sampling = 'log',
						 computed = False):
		"""
		Perform unsupervised clustering on acoustic feature data using UMAP for dimensionality reduction
		and HDBSCAN for clustering.

		This function loads acoustic feature files (assumed to be `.npz` format), extracts a subset
		of features from each, filters out invalid or short samples, and then applies dimensionality
		reduction followed by clustering. It stores the relevant data and results as instance attributes.

		Parameters
		----------
		nfeatures : int
			Number of frequency bins to keep (logarithmic scale) for clustering
		acoustic_path : str, optional
		    Path to the directory containing acoustic feature files. Each file is expected to contain 'spectro' and 'time' arrays, and 'len_spectro' as a scalar. Defaults to None.
		min_cluster_size : int, optional
		    The minimum size of clusters for HDBSCAN. Smaller clusters will be treated as noise. Default is 50.
		min_samples : int, optional
		    The number of samples in a neighborhood for a point to be considered a core point in HDBSCAN. Default is 10.
		sort : bool, optional
		    Whether or not to sort feature with increasing amplitude along temporal axis. Not implemented
		"""
		if not computed :
			fns = glob(os.path.join(acoustic_path, '*'))
			if freqs :
				frequencies = np.load(fns[0])['freq']
				self.posfeatures = np.array([np.argmin(abs(frequencies - elem)) for elem in freqs])
			elif (nfeatures <= 20) and (freq_sampling == 'log'):
				self.posfeatures = np.exp(np.arange(0, nfeatures) * np.log(513) / nfeatures).astype(int)
			else :
				self.posfeatures = np.arange(0, 512, max(1, 1//nfeatures)).astype(int)
			X = []
			_fns = []
			start, stop = [], []
			for fn in tqdm(fns):
				data = np.load(fn)
				if (data['len_spectro'] <= tmax*20) or (np.nanmax(data['depth']) < 50) :
					continue
				if not  bathy[0] <= np.nanmax(data['bathymetry']) <= bathy[-1]:
					continue
				_data = data['spectro'][int(tmin*20):int(tmax*20):timestep, self.posfeatures]
				if np.isnan(_data).sum() != 0 :
					continue
				_fns.append(fn)
				if sort :
					_data = sort_spectrogram(_data)
				X.append(_data)
				start.append(data['time'][0])
				stop.append(data['time'][-1])
			self.cluster_fns = np.array(_fns)
			X = np.array(X)
			self.X = X.reshape(X.shape[0], -1)
			self.start, self.stop = np.array(start), np.array(stop)
		project = umap.UMAP()
		self.embed = project.fit_transform(self.X)
		self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(self.embed)

	def save_cluster(self, cluster, overwrite = False, **metadata) :
		start = self.start[np.isin(self.clusterer.labels_, cluster)]
		stop = self.stop[np.isin(self.clusterer.labels_, cluster)]
		timestamps = self.ds['time'][:]
		drifts = np.zeros((len(timestamps)))
		for _start, _stop in zip(start, stop):
			drifts[(timestamps >= _start) & (timestamps <= _stop)] = 1
		metadata = {**{'method':'Clustering', 'features':f' Hz, '.join(self.posfeatures.astype(str))}, **metadata}
		self.create_variable('cluster_drifts', drifts, timestamps, overwrite, **metadata)

	def save_to_csv(self, path, name):
		dive_csv = pd.read_csv(path)
		embed0 = np.full(len(dive_csv), np.nan)
		embed1 = np.full(len(dive_csv), np.nan)
		cluster = np.full(len(dive_csv), np.nan)
		for i, fn in enumerate(self.cluster_fns) :
			dive_number = int(fn.split('.')[0][-4:])
			embed0[dive_number] = self.embed[i, 0]
			embed1[dive_number] = self.embed[i, 1]
			cluster[dive_number] = self.clusterer.labels_[i]
		if name + '_embed0' not in dive_csv.columns :
			dive_csv[name + '_embed0'] = embed0
			dive_csv[name + '_embed1'] = embed1
		dive_csv.to_csv(path, index=False)

	def acoustic_threshold(self, frequency : Union[List, int, str] = 'all', acoustic_path = None, N = 5, threshold = None):
		if not threshold :
			drifts = [[]*N]
			if frequency == 'all' :
				freq = slice(None)
			else :
				#freq = frequency if isinstance(self.depid, List) else [frequency]
				freq = frequency if isinstance(frequency, List) else [frequency]
			fns = sorted(glob(os.path.join(acoustic_path, '*')))
			avg_freq = np.nanmean(np.load(fns[10])['spectro'][:,freq], axis = 0)
			thresholds = np.linspace(np.nanmin(avg_freq), np.nanmax(avg_freq), N)
			for fn in tqdm(fns) :
				data = np.nanmean(np.load(fn)['spectro'][:,freq], axis = 1)
				for i, thresh in enumerate(thresholds) :
					_drift = np.zeros((len(data['len_spectro'])))
					_drift[data <= thresh] = 1
					drifts[i].extend(_drift)
			self.acoustic_drifts = np.array(drifts)
			self.thresholds = thresholds
		else :
			drifts = []
			if frequency == 'all' :
				freq = slice(None)
			else :
				freq = frequency if isinstance(self.depid, List) else [frequency]
			fns = glob(os.path.join(acoustic_path, '*'))
			for fn in tqdm(fns) :
				data = np.nanmean(np.load(fn)['spectro'][:,freq], axis = 1)
				_drift = np.zeros((len(data['len_spectro'])))
				_drift[data <= threshold] = 1
				drifts.extend(_drift)
			self.acoustic_drifts = np.array(drifts)

	def acoustic_feature_threshold(self, frequency : Union[List, int, str] = 'all', acoustic_path = None, N = 5, threshold = None, smoothing_length = 10, drift_length = 180):
		if not threshold :
			if frequency == 'all' :
				freq = slice(None)
			else :
				#freq = frequency if isinstance(self.depid, List) else [frequency]
				freq = frequency if isinstance(frequency, List) else [frequency]
			fns = sorted(glob(os.path.join(acoustic_path, '*')))
			avg_freq = np.nanmean(np.load(fns[10])['spectro'][:, freq], axis=0)
			thresholds = np.linspace(np.nanmin(avg_freq), np.nanmax(avg_freq), N)
			data = []
			for fn in tqdm(fns) :
				data.extend(np.nanmean(np.load(fn)['spectro'][:,freq], axis = 1))
			data = np.array(data)
			acoustic = np.full(len(self.ds['depth'][:]), np.nan)
			acoustic_bound = int(np.ceil(smoothing_length / 2 / self.ds.sampling_rate))
			pbar = tqdm(total=len(data) - 2 * acoustic_bound - 1, position=0, leave=True)
			pbar.set_description('Applying smoothing window')
			for j in range(acoustic_bound, len(data) - acoustic_bound - 1):
				#acoustic[j] = np.nanmean(data[j - acoustic_bound: j + acoustic_bound +1])
				acoustic[j] = np.nanmedian(data[j - acoustic_bound:j + acoustic_bound +1])
				pbar.update(1)
			# Find drift dives
			pbar = tqdm(total=len(thresholds), position=0, leave=True)
			pbar.set_description('Finding drift_dives')
			dive_type = np.full((len(thresholds), len(self.ds['depth'][:])), 0)
			for i, thresh in enumerate(thresholds) :
				drift_bound = int(np.ceil(drift_length / self.ds.sampling_rate / 2))
				for k in range(drift_bound, dive_type.shape[1] - drift_bound - 1):
					if np.all(acoustic[k - drift_bound: k + drift_bound] <= thresh) :
						dive_type[i, k - drift_bound: k + drift_bound] = 1
				pbar.update(1)
			self.acoustic_drifts = np.array(dive_type)
			self.thresholds = thresholds
		else :
			if frequency == 'all' :
				freq = slice(None)
			else :
				#freq = frequency if isinstance(self.depid, List) else [frequency]
				freq = frequency if isinstance(frequency, List) else [frequency]
			fns = glob(os.path.join(acoustic_path, '*'))
			data = []
			for fn in tqdm(fns) :
				data.extend(np.nanmean(np.load(fn)['spectro'][:,freq], axis = 1))
			data = np.array(data)

			acoustic = np.full(len(self.ds['depth'][:]), np.nan)
			acoustic_bound = int(np.ceil(smoothing_length / 2 / self.ds.sampling_rate))
			pbar = tqdm(total=len(data) - 2 * acoustic_bound - 1, position=0, leave=True)
			pbar.set_description('Applying smoothing window')
			for j in range(acoustic_bound, len(data) - acoustic_bound - 1):
				#acoustic[j] = np.nanmean(data[j - acoustic_bound: j + acoustic_bound + 1])
				acoustic[j] = np.nanmedian(data[j - acoustic_bound : j + acoustic_bound +1])
				pbar.update(1)
			dive_type = np.full(len(self.ds['depth'][:]), 0)
			drift_bound = int(np.ceil(drift_length / self.ds.sampling_rate / 2))
			pbar = tqdm(total=len(dive_type) - 2 * drift_bound - 1, position=0, leave=True)
			pbar.set_description('Finding drift_dives')
			for k in range(drift_bound, len(dive_type) - drift_bound - 1):
				if np.all(acoustic[k - drift_bound: k + drift_bound] <= threshold) :
					dive_type[k - drift_bound: k + drift_bound] = 1
				pbar.update(1)
			self.acoustic_drifts = np.array(dive_type)

	def compute_metric(self, cluster = [1]):
		acc_drift = self.ds['acc_drift'][:].data
		depth_drift = self.ds['depth_drift'][:].data
		dives = self.ds['dives'][:].data
		inertial = self.ds['inertial_drift'][:].data
		acc_dive, depth_dive, inert_dive, ac_dive = [], [], [], []
		for fn in self.cluster_fns:
			dive = int(fn.split('.')[0][-4:])
			if np.all(acc_drift[dives == dive] == 0) == False:
				acc_dive.append(1)
			else:
				acc_dive.append(0)
			if np.all(inertial[dives == dive] == 0) == False:
				inert_dive.append(1)
			else:
				inert_dive.append(0)
			if np.all(depth_drift[dives == dive] == 0) == False:
				depth_dive.append(1)
			else:
				depth_dive.append(0)
		label = np.array(acc_dive) & np.array(depth_dive)
		if 'clusterer' in dir(self):
			ac_dive = np.zeros(len(label))
			ac_dive[np.isin(self.clusterer.labels_, cluster)] = 1
		else :
			for fn in self.cluster_fns:
				dive = int(fn.split('.')[0][-4:])
				if np.all(self.ds['cluster_drifts'][:].data[dives == dive] == 0) == False:
					ac_dive.append(1)
				else:
					ac_dive.append(0)
		label = np.array(acc_dive) & np.array(depth_dive)
		conf1 =  confusion_matrix(label, ac_dive, normalize='true')
		conf2 = confusion_matrix(inert_dive, ac_dive, normalize='true')
		conf3 = confusion_matrix(label, inert_dive, normalize='true')
		f1 = f1_score(label, ac_dive)
		return conf1, conf2, conf3, f1




