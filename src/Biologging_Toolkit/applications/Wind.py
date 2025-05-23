from Biologging_Toolkit.wrapper import Wrapper
from Biologging_Toolkit.config.config_weather import *
from Biologging_Toolkit.utils.format_utils import *
from Biologging_Toolkit.processing.Dives import *
from Biologging_Toolkit.config.config_weather import *
import numpy as np
from typing import Union, Tuple, List
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import sklearn.metrics as metrics
import pandas as pd
import os
from glob import glob
from tqdm import tqdm


def beaufort(x):
    return next((i for i, limit in enumerate([0.3, 1.6, 3.4, 5.5, 8, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]) 
                 if x < limit), 12)

class Wind():

	seq_length = 800
	
	def __init__(
		self,
		depid : Union[str, List],
		*,
		path : Union[str, List] = None,
		acoustic_path : Union[str, List] = None,
		method: str = 'Pensieri',
		data : str = None,
		split_method : str = 'depid',
		nsplit : float = 0.8,
		test_depid : Union[str, List] = 'ml17_280a',
		ground_truth = 'wind_speed'
		):

		"""
		"""
		self.depid = depid
		self.path = path
		if acoustic_path :
			self.acoustic_path = [acoustic_path]
		else :
			self.acoustic_path = ['']*len(self.depid)
		if isinstance(self.depid, List) :
			assert len(self.depid) == len(self.acoustic_path) and len(self.depid) == len(self.path), "Please provide paths for each depid"
		else :
			self.depid = [self.depid]
			self.path = [self.path]

		self.ground_truth = ground_truth
		self.method = empirical[method]
		if data : 
			self.method['frequency'] = data
		self.ref = self.depid[0] if len(self.depid) == 1 else test_depid

		df = {'fns':[], 'dive':[], 'begin_time':[], 'end_time':[], 'depid':[], 'wind_speed':[], data:[]}
		for dep_path, dep, ac_path in zip(self.path, self.depid, self.acoustic_path) :
			_df = pd.read_csv(os.path.join(dep_path, f'{dep}_dive.csv'))
			_df['depid'] = dep
			for i, row in _df.iterrows() :
				if os.path.exists(os.path.join(ac_path, f'acoustic_dive_{int(row.dive):05d}.npz')):
					df['fns'].append(os.path.join(ac_path, f'acoustic_dive_{int(row.dive):05d}.npz'))
				else :
					df['fns'].append('N/A')
				df['dive'].append(row.dive)
				df['begin_time'].append(row.begin_time)
				df['end_time'].append(row.end_time)
				df['depid'].append(row.depid)
				df['wind_speed'].append(row[ground_truth])
				df[data].append(row[data]) if data else df[data].append(np.nan)
		self.df = pd.DataFrame(df)

		if split_method == 'depid':
			train_split, test_split = get_train_test_split(self.df.fns.to_numpy(), self.df.index.to_numpy(), self.df.depid.to_numpy(), method = split_method, test_depid = test_depid)
			self.train_split = train_split[1]
			self.test_split = test_split[1]
		else :
			self.train_split, self.test_split = get_train_test_split(self.df.fns.to_numpy(), self.df.index.to_numpy(), self.df.depid.to_numpy(), method = split_method, split = nsplit)
		self.popt, self.wind_model_stats = {}, {}

	def __str__(self):
		if 'wind_model_stats' in dir(self):
			print('Model has been trained with following parameters : \n')
			for key, value in self.method.items():
				print(f"{key} : {value}")
			print('ground truth : ', self.ground_truth)
			print('-----------\nThe model has the following performance :')
			for key, value in self.wind_model_stats.items():
				print(f"{key} : {value}")
			return "You can plot your estimation using the plot_estimation() method"
		else :
			print('Model has not been fitted to any data yet.\nWill be fitted with following parameters : \n')
			for key, value in self.method.items():
				print(f"{key:<{6}} : {value}")
			return "To fit your model, please call skf_fit() for example"
	

	def fetch_data(self, method = 'upwards', aggregation = 'mean', frequency = 5000):

		if aggregation == 'mean' :
			agg = np.nanmean
		elif aggregation == 'median' :
			agg = np.nanmedian
		elif aggregation == 'max':
			agg = lambda x : np.mean(np.sort(x)[-50:])
		for i, depid in enumerate(self.depid) :
			wind_speed = []
			spl = []
			dive = Dives(depid, path = self.path[i])
			if method == 'upwards':
				mask, down = dive.get_dive_direction(dive.ds['depth'][:].data[::20])
				mask = resample_boolean_array(mask, len(dive.ds['depth'][:]))
			elif method == 'downwards':
				up, mask = dive.get_dive_direction(dive.ds['depth'][:].data[::20])
				mask = resample_boolean_array(mask, len(dive.ds['depth'][:]))
			else :
				mask = True
			for j, row in dive.dive_ds.iterrows() :
				try :
					_data = np.load(os.path.join(self.acoustic_path[i],f'acoustic_dive_{int(row.dive):05d}.npz'))
				except FileNotFoundError :
					spl.append(np.nan)
					wind_speed.append(np.nan)
					continue
				idx_freq = np.argmin(abs(_data['freq'] - frequency))
				time_mask = dive.ds['time'][:].data[(dive.ds['dives'][:].data == row.dive) & mask & (dive.ds['depth'][:].data > 10)]
				spl.append(agg(_data['spectro'][np.isin(_data['time'], time_mask), idx_freq]))
				wind_speed.append(agg(dive.ds['wind_speed'][:].data[(dive.ds['dives'][:].data == row.dive) & mask]))
			dive.dive_ds['wind_speed'] = wind_speed
			dive.dive_ds[f'{method}_{aggregation}_{frequency}'] = spl
			dive.dive_ds.to_csv(dive.dive_path, index = None)
	
	def median_filtering(self, kernel_size = 5):
		'''
		Whether or not to apply scipy's median filtering to data
		'''
		self.df['filtered'] = medfilt(self.df[self.method['frequency']], kernel_size = kernel_size)
		self.method['frequency'] = 'filtered'

	def depid_fit(self, **kwargs) :
		default = {'scaling_factor':0.2, 'maxfev':25000}
		params = {**default, **kwargs}
		if 'bounds' not in params.keys():
			params['bounds'] = np.hstack((np.array([[value-params['scaling_factor']*abs(value), value+params['scaling_factor']*abs(value)] for value in self.method['parameters'].values()]).T, [[-np.inf],[np.inf]]))
		trainset = self.df.loc[self.train_split].dropna(subset = ['wind_speed', self.method['frequency']])
		testset = self.df.loc[self.test_split].dropna(subset = ['wind_speed', self.method['frequency']])
		popt, popv = curve_fit(self.method['function'], trainset[self.method['frequency']].to_numpy(), trainset['wind_speed'].to_numpy(), bounds = params['bounds'], maxfev=params['maxfev'])
		estimation = self.method['function'](testset[self.method['frequency']].to_numpy(), *popt)
		mae = metrics.mean_absolute_error(testset['wind_speed'], estimation)
		rmse = metrics.root_mean_squared_error(testset['wind_speed'], estimation)
		r2 = metrics.r2_score(testset['wind_speed'], estimation)
		var = np.var(abs(testset['wind_speed'])-abs(estimation))
		std = np.std(abs(testset['wind_speed'])-abs(estimation))
		self.df.loc[testset.index, 'depid_estimation'] = estimation
		self.popt.update({'depid_fit' : popt})
		self.wind_model_stats.update({'depid_mae':mae, 'depid_rmse':rmse, 'depid_r2':r2, 'depid_var':var, 'depid_std':std})

		
	def temporal_fit(self, **kwargs):

		default = {'split':0.8, 'scaling_factor':0.2, 'maxfev':25000}
		params = {**default, **kwargs}
		if 'bounds' not in params.keys():
			params['bounds'] = np.hstack((np.array([[value-params['scaling_factor']*abs(value), value+params['scaling_factor']*abs(value)] for value in self.method['parameters'].values()]).T, [[-np.inf],[np.inf]]))
		self.df['temporal_estimation'] = np.nan
		self.df = self.df.dropna(subset = [self.method['frequency'], 'wind_speed'])
		trainset = self.df.iloc[:int(params['split']*len(self.df))]
		testset = self.df.iloc[int(params['split']*len(self.df)):]
		popt, popv = curve_fit(self.method['function'], trainset[self.method['frequency']].to_numpy(), trainset['wind_speed'].to_numpy(), bounds = params['bounds'], maxfev=params['maxfev'])
		estimation = self.method['function'](testset[self.method['frequency']].to_numpy(), *popt)
		mae = metrics.mean_absolute_error(testset['wind_speed'], estimation)
		rmse = metrics.root_mean_squared_error(testset['wind_speed'], estimation)
		r2 = metrics.r2_score(testset['wind_speed'], estimation)
		var = np.var(abs(testset['wind_speed'])-abs(estimation))
		std = np.std(abs(testset['wind_speed'])-abs(estimation))
		self.df.loc[testset.index, 'temporal_estimation'] = estimation
		self.popt.update({'temporal_fit' : popt})
		self.wind_model_stats.update({'temporal_mae':mae, 'temporal_rmse':rmse, 'temporal_r2':r2, 'temporal_var':var, 'temporal_std':std})

	def skf_fit(self, **kwargs):
		'''
		Parameters :
			n_splits: Number of stratified K folds used for training, defaults to 5
			scaling_factor: Percentage of variability around initial parameters, defaults to 0.2
		'''
		default = {'n_splits':5, 'scaling_factor':0.2, 'maxfev':25000}
		params = {**default, **kwargs}
		if 'bounds' not in params.keys():
			params['bounds'] = np.hstack((np.array([[value-params['scaling_factor']*abs(value), value+params['scaling_factor']*abs(value)] for value in self.method['parameters'].values()]).T, [[-np.inf],[np.inf]]))
		popt_tot, popv_tot = [], []
		mae, rmse, r2, var, std = [], [], [], [], []
		self.df['classes'] = self.df['wind_speed'].apply(beaufort)
		self.df['skf_estimation'] = np.nan
		self.df = self.df.dropna(subset = [self.method['frequency'], 'wind_speed'])
		skf = StratifiedKFold(n_splits=params['n_splits'])
		for i, (train_index, test_index) in enumerate(skf.split(self.df[self.method['frequency']], self.df.classes)):
			trainset = self.df.iloc[train_index]
			testset = self.df.iloc[test_index]
			popt, popv = curve_fit(self.method['function'], trainset[self.method['frequency']].to_numpy(), 
						  trainset['wind_speed'].to_numpy(), bounds = params['bounds'], maxfev = params['maxfev'])
			popt_tot.append(popt)
			popv_tot.append(popv)
			estimation = self.method['function'](testset[self.method['frequency']].to_numpy(), *popt)
			mae.append(metrics.mean_absolute_error(testset['wind_speed'], estimation))
			rmse.append(metrics.root_mean_squared_error(testset['wind_speed'], estimation))
			r2.append(metrics.r2_score(testset['wind_speed'], estimation))
			var.append(np.var(abs(testset['wind_speed'])-abs(estimation)))
			std.append(np.std(abs(testset['wind_speed'])-abs(estimation)))
			self.df.loc[testset.index, 'skf_estimation'] = estimation
		self.popt.update({'skf_fit' : np.mean(popt_tot, axis=0)})
		self.wind_model_stats.update({'skf_mae':np.mean(mae), 'skf_rmse':np.mean(rmse), 'skf_r2':np.mean(r2), 'skf_var':np.mean(var), 'skf_std':np.mean(std)})


	def two_regime(self, func1, func2, bounds, threshold=0.17, variable=8000, train_method='skf_fit'):
		"""
		Method to apply a two regime model to acoustic data.
		func1 and bounds[0] are applied to data under the defined threshold quantile.
		func2 and bounds[1] are applied to data above the defined threshold quantile.
		train_method : 'skf_fit' or 'temporal_fit' depending on fitting protocol.
		"""
		_df = self.df
		met = train_method.split('_')[0]
		_df[f'{met}_estimation'] = np.nan
		_train_split = self.train_split
		_test_split = self.test_split
		idx = np.where(_df[variable] >= threshold)[0]

		self.train_split = list(set(_train_split) & set(idx))
		self.test_split = list(set(_test_split) & set(idx))
		self.method = func1
		if len(self.df.loc[self.test_split]) < 15:
			pass
		else:
			getattr(self, train_method)(bounds=bounds[0])
			_df[f'{met}_estimation'].loc[idx] = self.df[f'{met}_estimation']
			self.popt[f'{train_method}_1'] = self.popt[train_method]

		idx = np.where(_df[variable] < threshold)[0]
		self.train_split = list(set(_train_split) & set(idx))
		self.test_split = list(set(_test_split) & set(idx))
		self.method = func2
		if len(self.df.loc[self.test_split]) < 15:
			pass
		else:
			getattr(self, train_method)(bounds=bounds[1])
			_df[f'{met}_estimation'].loc[idx] = self.df[f'{met}_estimation']
			self.popt[f'{train_method}_2'] = self.popt[train_method]

		self.train_split = _train_split
		self.test_split = _test_split
		self.df = _df
		self.method["two_regime"] = [func1, func2]
		self.threshold = threshold
		print(f'Estimation contains {np.isnan(self.df[f"{met}_estimation"].to_numpy()).sum() / len(self.df)}% NaN values')
		self.wind_model_stats.update(
			{
				f"{met}_mae": np.nanmean(abs(self.df[self.ground_truth].to_numpy() - self.df[f'{met}_estimation'])),
				f"{met}_rmse": np.sqrt(np.nanmean((self.df[self.ground_truth].to_numpy() - self.df[f'{met}_estimation']) ** 2)),
				# f"{met}_r2": metrics.r2_score(self.df[self.ground_truth], self.df[f'{met}_estimation']),
				f"{met}_var": np.var(abs(self.df[self.ground_truth] - self.df[f'{met}_estimation'])),
				f"{met}_std": np.std(abs(self.df[self.ground_truth] - self.df[f'{met}_estimation']))
			})

