import pandas as pd
from typing import Union, Tuple, List
from Biologging_Toolkit.utils.format_utils import *
from scipy.optimize import curve_fit
import sklearn.metrics as metrics

class Rain():

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
		test_depid : Union[str, List] = 'ml17_280a'
		):

		"""
		"""
		self.depid = depid
		self.path = path
		if acoustic_path :
			self.acoustic_path = acoustic_path
		else :
			self.acoustic_path = ['']*len(self.depid)
		if isinstance(self.depid, List) :
			assert len(self.depid) == len(self.acoustic_path) and len(self.depid) == len(self.path), "Please provide paths for each depid"
		else :
			self.depid = [self.depid]
			self.path = [self.path]

		self.method = empirical[method]
		if data : 
			self.method['frequency'] = data
		self.ref = self.depid[0] if len(self.depid) == 1 else test_depid

		df = {'fns':[], 'dive':[], 'begin_time':[], 'end_time':[], 'depid':[], 'tp':[], data:[]}
		for dep_path, dep, ac_path in zip(self.path, self.depid, self.acoustic_path) :
			_df = pd.read_csv(os.path.join(dep_path, f'{dep}_dive.csv'))
			_df['depid'] = dep
			for i, row in _df.iterrows() :
				if os.path.exists(os.path.join(ac_path, f'{dep}_dive_{int(row.dive):05d}.npz')):
					df['fns'].append(os.path.join(ac_path, f'{dep}_dive_{int(row.dive):05d}.npz'))
				else :
					df['fns'].append('N/A')
				df['dive'].append(row.dive)
				df['begin_time'].append(row.begin_time)
				df['end_time'].append(row.end_time)
				df['depid'].append(row.depid)
				df['tp'].append(row.tp)
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
		
	def depid_fit(self, **kwargs) :
		default = {'scaling_factor':0.2, 'maxfev':25000}
		params = {**default, **kwargs}
		
		# if 'bounds' not in params.keys():
		# 	params['bounds'] = np.hstack((np.array([[value-params['scaling_factor']*abs(value), value+params['scaling_factor']*abs(value)] for value in self.method['parameters'].values()]).T, [[-np.inf],[np.inf]]))
	
		trainset = self.df.loc[self.train_split].dropna(subset = ['tp', self.method['frequency']])
		testset = self.df.loc[self.test_split].dropna(subset = ['tp', self.method['frequency']])
		
		popt, popv = curve_fit(self.method['function'], trainset[self.method['frequency']].to_numpy(), trainset['tp'].to_numpy(), bounds = params['bounds'], maxfev=params['maxfev'])
		estimation = self.method['function'](testset[self.method['frequency']].to_numpy(), *popt)
		
		mae = metrics.mean_absolute_error(testset['tp'], estimation)
		rmse = metrics.root_mean_squared_error(testset['tp'], estimation)
		r2 = metrics.r2_score(testset['tp'], estimation)
		var = np.var(abs(testset['tp'])-abs(estimation))
		std = np.std(abs(testset['tp'])-abs(estimation))
		
		self.df.loc[testset.index, 'depid_estimation'] = estimation
		self.popt.update({'depid_fit' : popt})
		self.wind_model_stats.update({'depid_mae':mae, 'depid_rmse':rmse, 'depid_r2':r2, 'depid_var':var, 'depid_std':std})