import torch
from torch import nn, tensor, utils, device, cuda, optim, long, save
from Biologging_Toolkit.wrapper import Wrapper
import numpy as np
from typing import Union, Tuple, List
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import sklearn.metrics as metrics
import pandas as pd
from glob import glob
from tqdm import tqdm


def beaufort(x):
    return next((i for i, limit in enumerate([0.3, 1.6, 3.4, 5.5, 8, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]) 
                 if x < limit), 12)

class Wind(Wrapper):
	
	
	def __init__(
		self,
		dataset_path: str,
		method: str = None,
		ground_truth : str = 'era',
		weather_params : dict = None,
		*,
		gps_coordinates: Union[str, List, Tuple, bool] = True,
		depth: Union[str, int, bool] = True,
		owner_group: str = None,
		batch_number: int = 5,
		local: bool = True,
		
		era : Union[str, bool] = False,
		annotation : Union[dict, bool] = False,
		other: dict = None
		):
		
		"""		
		Parameters:
			dataset_path (str): The path to the dataset.
			method (str) : Method or more generally processing pipeline and model used to estimate wind speed. Found in config_weather.py.
			ground_truth (str) : Column name from auxiliary data that stores wind speed data that will be used as ground truth. 
			dataset_sr (int, optional): The dataset sampling rate. Default is None.
			weather_params (dict) : Enter your own parameters for wind estimation. Will be taken into account if method = None.
				- frequency : 'int'        
				- samplerate : 'int'
				- preprocessing
					- nfft : 'int'
					- window_size : 'int'
					- spectro_duration : 'int'
					- window : 'str'
					- overlap : 'float'
				- function : func
				- averaging_duration : 'int'
				- parameters
					- a : 'float'
					- b : 'float'
					- ...
			analysis_params (dict, optional): Additional analysis parameters. Default is None.
			gps_coordinates (str, list, tuple, bool, optional): Whether GPS data is included. Default is True. If string, enter the filename (csv) where gps data is stored.
			depth (str, int, bool, optional): Whether depth data is included. Default is True. If string, enter the filename (csv) where depth data is stored.
			era (bool, optional): Whether era data is included. Default is False. If string, enter the filename (Network Common Data Form) where era data is stored.
			annotation (bool, optional): Annotation data is included. Dictionary containing key (column name of annotation data) and absolute path of csv file where annotation data is stored. Default is False. 
			other (dict, optional): Additional data (csv format) to join to acoustic data. Key is name of data (column name) to join to acoustic dataset, value is the absolute path where to find the csv. Default is None.
		"""
				
		if method :
			self.method = empirical[method]
			analysis_params['nfft'] = self.method['preprocessing']['nfft']
			analysis_params['window_size'] = self.method['preprocessing']['window_size']
			analysis_params['spectro_duration'] = self.method['preprocessing']['spectro_duration']
			dataset_sr = self.method['samplerate']
		else :
			self.method = weather_params
			
		super().__init__(dataset_path, gps_coordinates=gps_coordinates, depth=depth, dataset_sr=weather_params['samplerate'], 
				   owner_group=owner_group, analysis_params=weather_params['preprocessing'], batch_number=batch_number, local=local,
				   era = era, annotation=annotation, other=other)
		
		self.ground_truth = ground_truth		
		if self.ground_truth not in self.df :
			print(f"Ground truth data '{self.ground_truth}' was not found in joined dataframe.\nPlease call the correct joining method or automatic_join()")
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
	

	def median_filtering(self, kernel_size = 5):
		'''
		Whether or not to apply scipy's median filtering to data
		'''
		self.df['filtered'] = medfilt(self.df[self.method['frequency']], kernel_size = kernel_size)
		self.method['frequency'] = 'filtered'

	
	def temporal_fit(self, **kwargs):
		default = {'split':0.8, 'scaling_factor':0.2, 'maxfev':25000}
		params = {**default, **kwargs}
		if 'bounds' not in params.keys():
			params['bounds'] = np.hstack((np.array([[value-params['scaling_factor']*abs(value), value+params['scaling_factor']*abs(value)] for value in self.method['parameters'].values()]).T, [[-np.inf],[np.inf]]))
		self.df['temporal_estimation'] = np.nan
		trainset = self.df.iloc[:int(params['split']*len(self.df))].dropna(subset = [self.method['frequency'], self.ground_truth])
		testset = self.df.iloc[int(params['split']*len(self.df)):].dropna(subset = [self.method['frequency'], self.ground_truth])
		popt, popv = curve_fit(self.method['function'], trainset[self.method['frequency']].to_numpy(), trainset[self.ground_truth].to_numpy(), bounds = params['bounds'], maxfev=params['maxfev'])
		estimation = self.method['function'](testset[self.method['frequency']].to_numpy(), *popt)
		mae = metrics.mean_absolute_error(testset[self.ground_truth], estimation)
		rmse = metrics.root_mean_squared_error(testset[self.ground_truth], estimation)
		r2 = metrics.r2_score(testset[self.ground_truth], estimation)
		var = np.var(abs(testset[self.ground_truth])-abs(estimation))
		std = np.std(abs(testset[self.ground_truth])-abs(estimation))
		self.df.loc[testset.index, 'temporal_estimation'] = estimation
		self.popt['temporal_fit'] = popt
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
		self.df['classes'] = self.df[self.ground_truth].apply(beaufort)
		self.df['skf_estimation'] = np.nan
		skf = StratifiedKFold(n_splits=params['n_splits'])
		for i, (train_index, test_index) in enumerate(skf.split(self.df[self.method['frequency']], self.df.classes)):
			trainset = self.df.iloc[train_index].dropna(subset = [self.method['frequency'], self.ground_truth])
			testset = self.df.iloc[test_index].dropna(subset = [self.method['frequency'], self.ground_truth])
			popt, popv = curve_fit(self.method['function'], trainset[self.method['frequency']].to_numpy(), 
						  trainset[self.ground_truth].to_numpy(), bounds = params['bounds'], maxfev = params['maxfev'])
			popt_tot.append(popt)
			popv_tot.append(popv)
			estimation = self.method['function'](testset[self.method['frequency']].to_numpy(), *popt)
			mae.append(metrics.mean_absolute_error(testset[self.ground_truth], estimation))
			rmse.append(metrics.root_mean_squared_error(testset[self.ground_truth], estimation))
			r2.append(metrics.r2_score(testset[self.ground_truth], estimation))
			var.append(np.var(abs(testset[self.ground_truth])-abs(estimation)))
			std.append(np.std(abs(testset[self.ground_truth])-abs(estimation)))
			self.df.loc[testset.index, 'skf_estimation'] = estimation
		self.popt['skf_fit'] = np.mean(popt_tot, axis=0)
		self.wind_model_stats.update({'skf_mae':np.mean(mae), 'skf_rmse':np.mean(rmse), 'skf_r2':np.mean(r2), 'skf_var':np.mean(var), 'skf_std':np.mean(std)})


	def lstm_fit(self, seq_length = 10, **kwargs) :
		default = {'learning_rate':0.001,'epochs':75,'weight_decay':0.000,'hidden_dim':512, 'n_splits':5, 'n_cross_validation':1}
		params = {**default, **kwargs}
		self.df['classes'] = self.df[self.ground_truth].apply(beaufort)
		self.df['lstm_estimation'] = np.nan
		self.df.dropna(subset = [self.method['frequency'], self.ground_truth], inplace = True)
		skf = StratifiedKFold(n_splits=params['n_splits'])
		split = skf.split(self.df[self.method['frequency']], self.df.classes)
		for i in range(min(params['n_splits'], params['n_cross_validation'])):
			# set the device (CPU or GPU)
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			# create the model and move it to the specified device
			model = dl_utils.RNNModel(1, params['hidden_dim'], 1, 1)
			model.to(device)
			# create the loss function and optimizer
			criterion = nn.MSELoss()
			optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

			train_indices, test_indices = next(split)
			trainset, testset = self.df.loc[train_indices], self.df.loc[test_indices]
			train_loader = utils.data.DataLoader(dl_utils.Wind_Speed(trainset, self.method['frequency'], self.ground_truth, seq_length=seq_length), batch_size = 64, shuffle = True)
			test_loader = utils.data.DataLoader(dl_utils.Wind_Speed(testset, self.method['frequency'], self.ground_truth, seq_length=seq_length), batch_size = 64, shuffle=False)
			estimation = dl_utils.train_rnn(model, train_loader, test_loader, criterion, optimizer, num_epochs = params['epochs'], device = device)		
			self.df.loc[test_indices, 'lstm_estimation'] = estimation

