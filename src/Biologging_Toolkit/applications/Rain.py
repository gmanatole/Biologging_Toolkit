from Biologging_Toolkit.wrapper import Wrapper
from Biologging_Toolkit.config.config_weather import *
from Biologging_Toolkit.utils.format_utils import *
from Biologging_Toolkit.processing.Dives import *
import numpy as np
from typing import Union, Tuple, List
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import sklearn.metrics as metrics
import pandas as pd
from glob import glob
from tqdm import tqdm
from sympy import Symbol,expand

def rain_scale(x):
    return next((i for i, limit in enumerate([0.3, 1.0, 4.0, 16.0, 50.0]) if x < limit), 5)


class Rain():

	seq_length = 800
	
	def __init__(
		self,
		depid : Union[str, List],
		*,
		path : Union[str, List] = None,
		acoustic_path : Union[str, List] = None,
		method: str = 'Nystuen',
		data : str = None,        
		split_method : str = 'random_split',
		nsplit : float = 0.8,
		test_depid : Union[str, List] = 'ml17_280a',
		df_data = "csv",
		empirical_offset = 1.25,
		optimized_thresh = True,
		frequency = 5000,
		wind_thresh = 7,
		rain_thresh = 0.1,
		remove_wind = False,
		wind_speed = 'wind_speed'
		):

		"""
		"""
		
		self.depid = depid
		self.path = path
		self.split_method = split_method
		self.test_depid = test_depid
		self.ground_truth = "IMERG (GPM NASA)"
		self.wind_speed = wind_speed
		self.nsplit = nsplit
		self.empirical_offset = empirical_offset
		self.optimized_thresh = optimized_thresh
		self.wind_thresh = wind_thresh
		self.rain_thresh = rain_thresh
		self.frequency = frequency

		if acoustic_path :
			self.acoustic_path = acoustic_path
		else :
			self.acoustic_path = ['']*len(self.depid)
		if isinstance(self.depid, List) :
			assert len(self.depid) == len(self.acoustic_path) and len(self.depid) == len(self.path), "Please provide paths for each depid"
		else :
			self.depid = [self.depid]
			self.path = [self.path]
			self.acoustic_path = [self.acoustic_path]

		self.method_name = method
		self.method = empirical_rain[method]
		if data : 
			self.method['frequency'] = data
		self.ref = self.depid[0] if len(self.depid) == 1 else test_depid

		self.create_df(df_data)

		if remove_wind :
			self.df = self.df[self.df.wind_speed < wind_thresh]
		self.calculate_and_add_slope(2000,8000)
		self.calculate_and_add_slope(8000,15000)

	def forward(self):
		conditions = [
			(self.df["precipitation_GPM"] > self.rain_thresh) & (self.df[self.wind_speed] < self.wind_thresh),
			(self.df["precipitation_GPM"] > self.rain_thresh) & (self.df[self.wind_speed] >= self.wind_thresh),
		]
		choices = ["R", "WR"]
		self.df["weather"] = np.select(conditions, choices, default="N")

		if self.split_method == 'depid' :
			self.df["Rain_Type_preds"] = np.nan
			for _depid in self.depid :
				train_split, test_split = get_train_test_split(self.df.fns.to_numpy(), self.df.index.to_numpy(), self.df.depid.to_numpy(), method = self.split_method, test_depid = _depid)
				self.train_split = train_split[1]
				self.test_split = test_split[1]
				self.classify_rain()
		else :
			self.train_split = list(self.df.index)
			self.test_split = list(self.df.index)
			self.classify_rain()

		self.df_r = self.df.loc[self.df["Rain_Type_preds"]=="R"].copy()
		self.df_r = self.df_r.dropna(subset = ['precipitation_GPM', f"upwards_mean_{self.frequency}"])
		self.df_r = self.df_r.reset_index(drop=True)

		if self.split_method == 'depid':
			train_split, test_split = get_train_test_split(self.df_r.fns.to_numpy(), self.df_r.index.to_numpy(), self.df_r.depid.to_numpy(), method = self.split_method, test_depid = self.test_depid)
			self.train_split = train_split[1]
			self.test_split = test_split[1]
		else :
			self.train_split, self.test_split = get_train_test_split(self.df_r.fns.to_numpy(), self.df_r.index.to_numpy(), self.df_r.depid.to_numpy(), method = self.split_method, split = self.nsplit)
		self.popt, self.rain_model_stats = {}, {}


	def classify_rain(self):
		combinations = [("upwards_mean_5000","upwards_mean_15000"),("upwards_mean_8000","upwards_mean_15000"),("upwards_mean_8000","slope_8000_15000")]
		coefs = []
		trainset = self.df.loc[self.train_split]
		for comb in combinations:
			polydeg = 1
			quant_val = 0.8
			quantile_df = pd.DataFrame({})
			for bin in range(-55, -25, 5):
				_df = trainset[(trainset[comb[0]] > bin) & (trainset[comb[0]] < bin + 5)]
				threshold = _df[comb[1]].quantile(quant_val)
				_df.loc[_df[comb[1]] > threshold, comb[1]] = np.nan
				_df = _df.dropna(subset=[comb[0], comb[1]])
				quantile_df = pd.concat([quantile_df, _df])
			model = np.poly1d(np.polyfit(quantile_df[comb[0]], quantile_df[comb[1]], polydeg))
			x_Symb = Symbol(comb[1])
			x_vals = np.linspace(trainset[comb[1]].min(), trainset[comb[1]].max(), 100)
			y_vals = model(x_vals)
			a, b = model.coefficients
			coefs.append((comb[0], comb[1], a, b))

		testset = self.df.loc[self.test_split]
		if self.optimized_thresh:
			conditions = [
				(testset[coefs[0][1]]>coefs[0][2]*testset[coefs[0][0]] +coefs[0][3]+self.empirical_offset/2) &
				(testset[coefs[1][1]]>coefs[1][2]*testset[coefs[1][0]] +coefs[1][3]+self.empirical_offset) &
				(testset[coefs[2][1]]>coefs[2][2]*testset[coefs[2][0]] +coefs[2][3]+self.empirical_offset/4)
			]
		else :
			conditions = [
				(testset[coefs[0][1]]>coefs[0][2]*testset[coefs[0][0]] +coefs[0][3]+self.empirical_offset) &
				(testset[coefs[1][1]]>coefs[1][2]*testset[coefs[1][0]] +coefs[1][3]+self.empirical_offset) &
				(testset[coefs[2][1]]>coefs[2][2]*testset[coefs[2][0]] +coefs[2][3]+self.empirical_offset)
			]

		choices = ["R"]
		self.df.loc[self.test_split, "Rain_Type_preds"] = np.select(conditions, choices, default="N")

		#conditions = [
		#	(self.df["precipitation_GPM"] > 0.1) & (self.df["wind_speed"] < self.wind_thresh),
		#]
		#choices = ["R"]

		#self.df["Rain_Type"] = np.select(conditions, choices, default="N+WR")

	def calculate_and_add_slope(self, freq1, freq2):
		slope_column_name = f"slope_{freq1}_{freq2}"
		spl1 = self.df[f'upwards_mean_{freq1}']
		spl2 = self.df[f'upwards_mean_{freq2}']
		delta_spl = spl2 - spl1
		delta_log_freq = np.log10(freq2) - np.log10(freq1)
		slope = delta_spl / delta_log_freq
		self.df[slope_column_name] = slope 
	
	def create_df(self, df_data):
		if df_data=="csv" :
			fns = []
			_dep = []
			df = pd.DataFrame({})
			for p, dep, ac_path in zip(self.path, self.depid, self.acoustic_path):
				df_csv = pd.read_csv(f"{p}/{dep}_dive.csv")
				df = pd.concat([df, df_csv])
				for i, row in df_csv.iterrows() :
					_dep.append(dep)
					if os.path.exists(os.path.join(ac_path, f'{dep}_dive_{int(row.dive):05d}.npz')):
						fns.append(os.path.join(ac_path, f'{dep}_dive_{int(row.dive):05d}.npz'))
					else :
						fns.append('N/A')
			df['fns']=fns
			df["depid"]=_dep

		elif df_data=="npz_reduced" : 
			df = pd.DataFrame({})
			for p, dep in zip(self.path, self.depid):
				print(dep)
				df_csv = pd.read_csv(f"{p}/{dep}_dive.csv")
				rows = []
				for idx, dive in tqdm(df_csv.iterrows(), total=len(df_csv)):

					npz_path = os.path.join(p, "dives", f"acoustic_dive_{idx:05d}.npz")
					npz_data = np.load(npz_path)
					
					time = npz_data["time"]
					precip = dive["precipitation_GPM"]
					ws = dive[self.wind_speed]
					upwards_mean_15000 = npz_data["spectro"].T[480]
					upwards_mean_8000 = npz_data["spectro"].T[256]
					upwards_mean_5000 = npz_data["spectro"].T[160]
					upwards_mean_2000 = npz_data["spectro"].T[64]
					for i in range(len(time)):
						rows.append({
							"fns":npz_path,
							"begin_time": time[i],
							"precipitation_GPM": float(precip),
							"wind_speed": float(ws),
							"upwards_mean_15000": upwards_mean_15000[i],
							"upwards_mean_8000": upwards_mean_8000[i],
							"upwards_mean_5000": upwards_mean_5000[i],
							"upwards_mean_2000": upwards_mean_2000[i],
							"depid":dep
						})

				df = pd.concat([df,pd.DataFrame(rows)])
		elif df_data=="npz":
			rows=[]
			for p, dep in zip(self.path, self.depid):
				print(dep)
				df_csv = pd.read_csv(f"{p}/{dep}_dive.csv")
				rows = []
				for idx, row in tqdm(df_csv.iterrows(), total=len(df_csv)):
					npz_path = os.path.join(p,dep,"dives",f'acoustic_dive_{int(row["dive"]):05d}.npz')
					npz = np.load(npz_path)
					freq, spectro = npz["freq"], npz["spectro"]
					
					mask = npz["depth"] > 10
					spectro_masked = spectro[mask] 

					target_length = 600
					current_length = spectro_masked.shape[0]

					if current_length < target_length:
						pad_width = ((0, target_length - current_length), (0, 0)) 
						padded_spectro = np.pad(spectro_masked, pad_width, mode='constant')
					else:
						padded_spectro = spectro_masked[:target_length, :]

					flat_spectro = padded_spectro.flatten()
					row_npz = np.insert(flat_spectro, 0, row["precipitation_GPM"])
					rows.append(row_npz)
					
			final_array = np.stack(rows)
			df = pd.DataFrame(final_array)
		else :
			raise Exception("/!\ df_data must be 'csv' or 'npz") 
		
		self.df = df.reset_index()
	
	def estimate_bibliography(self, offset=100):
		estimation = self.method['function'](self.df_r[self.method['frequency']],self.method['parameters']["a"],self.method['parameters']["b"],offset)
		mae = metrics.mean_absolute_error(self.df_r['precipitation_GPM'], estimation)
		rmse = metrics.root_mean_squared_error(self.df_r['precipitation_GPM'], estimation)
		r2 = metrics.r2_score(self.df_r['precipitation_GPM'], estimation)
		var = np.var(abs(self.df_r['precipitation_GPM'])-abs(estimation))
		std = np.std(abs(self.df_r['precipitation_GPM'])-abs(estimation))
		cc = np.corrcoef(self.df_r['precipitation_GPM'], estimation)[0][1]

		self.df_r.loc[self.df_r.index, 'depid_estimation'] = estimation
		self.rain_model_stats.update({'biblio_mae':mae, 'biblio_rmse':rmse, 'biblio_r2':r2, 'biblio_var':var, 'biblio_std':std,'biblio_cc':cc})
		
	def __str__(self):
		if 'rain_model_stats' in dir(self):
			print('Model has been trained with following parameters : \n')
			for key, value in self.method.items():
				print(f"{key} : {value}")
			print('ground truth : ', self.ground_truth)
			print('-----------\nThe model has the following performance :')
			for key, value in self.rain_model_stats.items():
				print(f"{key} : {value}")
			return""# "You can plot your estimation using the following methods : \nR_Utils.plot_rain_estimation(inst)\nR_Utils.plot_rain_estimation_cumulated(inst)"
		else :
			print('Model has not been fitted to any data yet.\nWill be fitted with following parameters : \n')
			for key, value in self.method.items():
				print(f"{key:<{6}} : {value}")
			return "To fit your model, please call skf_fit() for example"
	
	def depid_fit(self, **kwargs) :
		default = {'scaling_factor':0.2, 'maxfev':25000}
		params = {**default, **kwargs}
		if 'bounds' not in params.keys():
			params['bounds'] = np.hstack((np.array([[value-params['scaling_factor']*abs(value), value+params['scaling_factor']*abs(value)] for value in self.method['parameters'].values()]).T, [[-np.inf],[np.inf]]))
		
		if 'Rain_Type_preds' not in self.df_r.columns :
			raise Exception("You must use self.classify() first before trying to estimate") 
	
		trainset = self.df_r.loc[self.train_split]
		testset = self.df_r.loc[self.test_split]
		popt, popv = curve_fit(self.method['function'], trainset[f"upwards_mean_{self.frequency}"].to_numpy(), trainset['precipitation_GPM'].to_numpy(), bounds = params['bounds'], maxfev=params['maxfev'])
		estimation = self.method['function'](testset[f"upwards_mean_{self.frequency}"].to_numpy(), *popt)

		mae = metrics.mean_absolute_error(testset['precipitation_GPM'], estimation)
		rmse = metrics.root_mean_squared_error(testset['precipitation_GPM'], estimation)
		r2 = metrics.r2_score(testset['precipitation_GPM'], estimation)
		var = np.var(abs(testset['precipitation_GPM'])-abs(estimation))
		std = np.std(abs(testset['precipitation_GPM'])-abs(estimation))
		cc = np.corrcoef(testset['precipitation_GPM'], estimation)[0][1]

		self.df_r.loc[testset.index, 'depid_estimation'] = estimation
		self.popt.update({'depid_fit' : popt})
		self.rain_model_stats.update({'depid_mae':mae, 'depid_rmse':rmse, 'depid_r2':r2, 'depid_var':var, 'depid_std':std,'depid_cc':cc})
	
	def temporal_fit(self, **kwargs):

		default = {'split':0.8, 'scaling_factor':0.2, 'maxfev':25000}
		params = {**default, **kwargs}
		if 'bounds' not in params.keys():
			params['bounds'] = np.hstack((np.array([[value-params['scaling_factor']*abs(value), value+params['scaling_factor']*abs(value)] for value in self.method['parameters'].values()]).T, [[-np.inf],[np.inf]]))
		self.df_r['temporal_estimation'] = np.nan
		self.df_r = self.df_r.dropna(subset = [f"upwards_mean_{self.frequency}", 'precipitation_GPM'])
		
		trainset = self.df_r.iloc[:int(params['split']*len(self.df_r))]
		testset = self.df_r.iloc[int(params['split']*len(self.df_r)):]
		self.train_split = trainset.index.to_numpy()
		self.test_split = testset.index.to_numpy()

		popt, popv = curve_fit(self.method['function'], trainset[f"upwards_mean_{self.frequency}"].to_numpy(), trainset['precipitation_GPM'].to_numpy(), bounds = params['bounds'], maxfev=params['maxfev'])
		estimation = self.method['function'](testset[f"upwards_mean_{self.frequency}"].to_numpy(), *popt)
		
		mae = metrics.mean_absolute_error(testset['precipitation_GPM'], estimation)
		rmse = metrics.root_mean_squared_error(testset['precipitation_GPM'], estimation)
		r2 = metrics.r2_score(testset['precipitation_GPM'], estimation)
		var = np.var(abs(testset['precipitation_GPM'])-abs(estimation))
		std = np.std(abs(testset['precipitation_GPM'])-abs(estimation))
		cc = np.corrcoef(testset['precipitation_GPM'], estimation)[0][1]

		self.df_r.loc[testset.index, 'temporal_estimation'] = estimation
		self.popt.update({'temporal_fit' : popt})
		self.rain_model_stats.update({'temporal_mae':mae, 'temporal_rmse':rmse, 'temporal_r2':r2, 'temporal_var':var, 'temporal_std':std,'temporal_cc':cc})

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
		self.df_r['classes'] = self.df_r['precipitation_GPM'].apply(rain_scale)
		self.df_r['skf_estimation'] = np.nan
		self.df_r = self.df_r.dropna(subset = [f"upwards_mean_{self.frequency}", 'precipitation_GPM'])
		skf = StratifiedKFold(n_splits=params['n_splits'])
		for i, (train_index, test_index) in enumerate(skf.split(self.df_r[f"upwards_mean_{self.frequency}"], self.df_r.classes)):
			trainset = self.df_r.iloc[train_index]
			testset = self.df_r.iloc[test_index]
			popt, popv = curve_fit(self.method['function'], trainset[f"upwards_mean_{self.frequency}"].to_numpy(), 
						  trainset['precipitation_GPM'].to_numpy(), bounds = params['bounds'], maxfev = params['maxfev'])
			popt_tot.append(popt)
			popv_tot.append(popv)
			estimation = self.method['function'](testset[f"upwards_mean_{self.frequency}"].to_numpy(), *popt)
			mae.append(metrics.mean_absolute_error(testset['precipitation_GPM'], estimation))
			rmse.append(metrics.root_mean_squared_error(testset['precipitation_GPM'], estimation))
			r2.append(metrics.r2_score(testset['precipitation_GPM'], estimation))
			var.append(np.var(abs(testset['precipitation_GPM'])-abs(estimation)))
			std.append(np.std(abs(testset['precipitation_GPM'])-abs(estimation)))
			cc = np.corrcoef(testset['precipitation_GPM'], estimation)[0][1]
			self.df_r.loc[testset.index, 'skf_estimation'] = estimation
		self.popt.update({'skf_fit' : np.mean(popt_tot, axis=0)})
		self.rain_model_stats.update({'skf_mae':np.mean(mae), 'skf_rmse':np.mean(rmse), 'skf_r2':np.mean(r2), 'skf_var':np.mean(var), 'skf_std':np.mean(std), 'skf_cc':np.mean(cc)})



