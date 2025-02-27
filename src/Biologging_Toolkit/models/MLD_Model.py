import os
import pandas as pd
import numpy as np
from datetime import datetime
from Biologging_Toolkit.utils.mixed_layer_depth_utils import *
from Biologging_Toolkit.utils.machine_learning_utils import MLDDataLoader, RegressionDataAugmenter, MLP, CNN, CNN_LSTM
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from typing import Union, List
from pygam import LinearGAM, GAM
from tqdm import tqdm
from torch import nn, utils
import torch
from src.Biologging_Toolkit.utils.machine_learning_utils import WeightedMSELoss


class MLDModel():
    """docstring for MLD_Model"""
    def __init__(self,
                 path,
                 depids,
                 test_depid : Union[List, str] = None,
                 wind = 'lstm',
                 params = ['peaks'],
                 target = 'mld',
                 smoothing = False,
                 structure = 'gust',
                 norm = False,
                 deepening = True,
                 find_peaks_params = {}):
        self.depids = depids.copy()
        if test_depid is None:
            test_depid = self.depids
        self.test_depid = [test_depid] if isinstance(test_depid, str) else test_depid
        self.path = path
        self.wind = wind
        self.fp_params = {**{'prominence':0.9, 'height':6, 'distance':3}, **find_peaks_params}
        self.params = params.copy()
        self.target = target
        self.smoothing = smoothing
        self.structure = structure
        self.deepening = deepening
        self.norm = norm
        self.params.append('const')
        if np.isin(self.target, self.params) :
            raise AttributeError('Your target variable is in your set of parameters')

    def construct_1D_structure(self, time_diff = 15):
        self.df = pd.DataFrame()
        for depid in self.depids :
            path = os.path.join(self.path, depid, f'{depid}_dive.csv')
            _df = create_gust_dataframe(depid, path, time_diff=time_diff, smoothing=self.smoothing, structure=self.structure)
        self.df = pd.concat((self.df, _df))
        self.df.reset_index(inplace = True, drop = True)
        self.df = self.df[self.df.var_mld != 0]
        self.df['mld_diff'] = self.df['mld'] - self.df['previous_mld']
        if self.deepening :
            self.df = self.df[self.df.mld_diff > 0]
        self.df.dropna(subset = self.params[:-1] + [self.target], inplace = True)
        if self.norm :
            for val in self.df.columns :
                if val == 'depid':
                    continue
                if self.norm == 'energy':
                    self.df[val] = (self.df[val].to_numpy() - np.nanmean(self.df[val].to_numpy())) / np.nanstd(self.df[val].to_numpy())
                else :
                    self.df[val] = (self.df[val].to_numpy() - np.nanmin(self.df[val].to_numpy())) / (np.nanmax(self.df[val].to_numpy()) - np.nanmin(self.df[val].to_numpy()))

    def construct_2D_structure(self, t0 = 15, t1 = 25, filter=5, size = 40):
        data = {'mld':[], 'wind':[], 'temp':[], 'previous_mld':[], 'gradient':[], 'density':[], 'depid':[], 'lat':[], 'lon':[], 'time':[]}
        for depid in self.depids :
            path = os.path.join(self.path, depid, f'{depid}_dive.csv')
            _mld, _previous_mld, _wind, _temp, _density, _gradient, _lat, _lon, _time = get_profiles(depid, path, data=self.wind, t0=t0, t1=t1, filter=filter, size=size, norm=self.norm)
            data['mld'].extend(_mld)
            data['wind'].extend(_wind)
            data['temp'].extend(_temp)
            data['previous_mld'].extend(_previous_mld)
            data['gradient'].extend(_gradient)
            data['density'].extend(_density)
            data['lat'].extend(_lat)
            data['lon'].extend(_lon)
            data['time'].extend(_time)
            data['depid'].extend([depid]*len(_mld))
        data['depid_id'] = np.tile(np.array((pd.Categorical(data['depid']).codes)/12), size).reshape(np.array(data['wind']).shape)
        data['hour'] = np.array([datetime.fromtimestamp(elem).hour for elem in np.array(data['time']).flatten()]).reshape(np.array(data['wind']).shape)
        data['norm_previous_mld'] = np.tile(
            np.expand_dims(((data['previous_mld'] - np.nanmin(data['previous_mld'])) / (np.nanmax(data['previous_mld']) - np.nanmin(data['previous_mld']))), axis = 1), reps = size)
        data['mld_diff'] = np.array(data['mld']).flatten() - np.array(data['previous_mld']).flatten()
        for key in data.keys():
            data[key] = np.array(data[key])
            if self.deepening :
                data[key] = data[key][data['mld_diff'] >= 0]
        self.data = data
        self.size = size

    def ols_regression(self) :
        if 'OLS_pred' not in self.df.columns:
            self.df['OLS_pred'] = np.nan
        self.test_df = self.df[np.isin(self.df.depid, self.test_depid)]
        self.train_df = self.df[~np.isin(self.df.depid, self.test_depid)]
        X = self.train_df[self.params[:-1]]
        X = sm.add_constant(X)
        y = self.train_df[self.target]
        self.ols = sm.OLS(y, X).fit()
        X_test = self.test_df[self.params[:-1]]
        X_test = sm.add_constant(X_test, has_constant = 'add')
        self.df.loc[self.test_df.index, 'OLS_pred'] = self.ols.predict(X_test)
        self.OLS_r_squared = self.ols.rsquared

    def glm_regression(self):
        if 'GLM_pred' not in self.df.columns:
            self.df['GLM_pred'] = np.nan
        self.test_df = self.df[np.isin(self.df.depid, self.test_depid)]
        self.train_df = self.df[~np.isin(self.df.depid, self.test_depid)]
        X = self.train_df[self.params[:-1]]
        X = sm.add_constant(X)
        y = self.train_df[self.target]
        self.glm = sm.GLM(y, X, family = sm.families.Poisson()).fit()
        X_test = self.test_df[self.params[:-1]]
        X_test = sm.add_constant(X_test, has_constant = 'add')
        self.df.loc[self.test_df.index, 'GLM_pred'] = self.glm.predict(X_test)
        try :
            self.GLM_r_squared = r2_score(self.test_df[self.target], self.df.loc[self.test_df.index, 'GLM_pred'])
        except :
            self.GLM_r_squared = np.nan

    def gls_regression(self) :
        if 'GLS_pred' not in self.df.columns:
            self.df['GLS_pred'] = np.nan
        self.test_df = self.df[np.isin(self.df.depid, self.test_depid)]
        self.train_df = self.df[~np.isin(self.df.depid, self.test_depid)]
        X = self.train_df[self.params[:-1]]
        X = sm.add_constant(X)
        y = self.train_df[self.target]
        ols_resid = sm.OLS(y, X).fit().resid
        res_fit = sm.OLS(np.asarray(ols_resid[1:]), np.asarray(ols_resid[:-1])).fit()
        rho = res_fit.params
        order = toeplitz(np.arange(len(X)))
        sigma = rho ** order
        self.gls = sm.GLS(y, X, sigma=sigma).fit()
        X_test = self.test_df[self.params[:-1]]
        X_test = sm.add_constant(X_test, has_constant = 'add')
        self.df.loc[self.test_df.index, 'GLS_pred'] = self.gls.predict(X_test)
        self.GLS_r_squared = self.gls.rsquared

    def random_forest(self, plot = False):
        if 'RF_pred' not in self.df.columns:
            self.df['RF_pred'] = np.nan
        self.test_df = self.df[np.isin(self.df.depid, self.test_depid)]
        self.train_df = self.df[~np.isin(self.df.depid, self.test_depid)]
        X = self.train_df[self.params[:-1]]
        y = self.train_df[self.target]
        self.rf = RandomForestRegressor()
        self.rf.fit(X, y)
        self.RF_importances = pd.Series(self.rf.feature_importances_, index=X.columns)
        if plot :
            self.RF_importances.sort_values(ascending=False).plot(kind='bar')
        X_test = self.test_df[self.params[:-1]]
        try :
            self.df.loc[self.test_df.index, 'RF_pred'] = self.rf.predict(X_test)
            self.RF_r_squared = self.rf.score(X_test, self.test_df[self.target])
        except ValueError :
            pass

    def generalized_additive_model(self, summary = False, **kwargs):
        default = {'distribution':'poisson','link':'log'}
        params = {**default, **kwargs}
        if 'GAM_pred' not in self.df.columns:
            self.df['GAM_pred'] = np.nan
        self.test_df = self.df[np.isin(self.df.depid, self.test_depid)]
        self.train_df = self.df[~np.isin(self.df.depid, self.test_depid)]
        X = self.train_df[self.params[:-1]]
        y = self.train_df[self.target]
        X = X.dropna()
        y = y.dropna()
        self.gam = GAM(link=params['link'], distribution=params['distribution']).fit(X.to_numpy(), y.to_numpy())
        if summary :
            self.gam.summary()
        X_test = self.test_df[self.params[:-1]]
        self.df.loc[self.test_df.index, 'GAM_pred'] = self.gam.predict(X_test)
        self.GAM_r_squared = r2_score(self.test_df[self.target], self.df.loc[self.test_df.index, 'GAM_pred'])

    def neural_network(self, nepoch = 15, model_type = 'MLP', augment = True, **kwargs) :
        default = {'input_size':120, 'learning_rate':0.001, 'weight_decay':0, 'batch_size':32, 'n_filters':5, 'kernel_size':5}
        params = {**default, **kwargs}
        if model_type == 'MLP' :
            model = MLP(params['input_size'])
        elif model_type == 'CNN' :
            model = CNN(1, params['n_filters'], params['kernel_size'], ninputs = len(self.params[:-1]), size = self.size-5)
        elif model_type == 'CNN_LSTM' :
            model = CNN_LSTM(1, params['n_filters'], params['kernel_size'], ninputs = len(self.params[:-1]), size = self.size-5)
        estimations, ground_truth = [], []
        self.X = np.hstack([self.data[key][self.data['depid'] != self.test_depid] for key in self.params[:-1]])
        self.x_test = np.hstack([self.data[key][self.data['depid'] == self.test_depid] for key in self.params[:-1]])
        self.Y = self.data[self.target][self.data['depid'] != self.test_depid]
        self.y_test = self.data[self.target][self.data['depid'] == self.test_depid]
        self.X = self.X[~np.isnan(self.Y)]
        self.Y = self.Y[~np.isnan(self.Y)]
        self.x_test = self.x_test[~np.isnan(self.y_test)]
        self.y_test = self.y_test[~np.isnan(self.y_test)]
        self.X = np.nan_to_num(self.X)
        self.x_test = np.nan_to_num(self.x_test)
        if augment :
            self.X, self.Y = RegressionDataAugmenter(self.X, self.Y).augment_data()
        self.losses = []
        trainloader = utils.data.DataLoader(MLDDataLoader(self.X, self.Y, len(self.params[:-1]), model_type), params['batch_size'], shuffle=True)
        testloader = utils.data.DataLoader(MLDDataLoader(self.x_test, self.y_test, len(self.params[:-1]), model_type), params['batch_size'], shuffle=False)
        #criterion = nn.MSELoss()
        criterion = WeightedMSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        pbar = tqdm(np.arange(0,nepoch,1), leave = True, position = 0)
        for epoch in pbar:
            for batch in trainloader:
                optimizer.zero_grad()
                input, labels = batch
                outputs = model(input)
                loss = criterion(outputs.reshape(len(outputs),1), labels.reshape(len(labels),1))
                loss.backward()
                if epoch == 1:
                    self.losses.append(loss.item())
                optimizer.step()
            model.eval()
            for batch in testloader:
                input, labels = batch
                outputs = model(input)
                if epoch == nepoch-1:
                    estimations.extend(outputs.detach().numpy().flatten())
                    ground_truth.extend(labels)
            pbar.update(1)
            pbar.set_description(f'R2 : {r2_score(labels, outputs.detach().numpy().flatten())}')
            model.train()
        self.ground_truth = np.array(ground_truth)
        self.neural_network_estimation = np.array(estimations)

    def temporal_linear_regression(self, time_diff = 15, tmax = 48, model = 'OLS'):
        model = model.upper()
        self.time_diff = time_diff
        r_squared, mae = [], []
        res_values = {par: [[], []] for par in self.params}
        for j in tqdm(np.arange(1,tmax,1)) :
            self.create_gust_dataframe(time_diff = j)
            if model == 'OLS':
                self.ols_regression()
            elif model == 'GLS' :
                self.gls_regression()
            if self.time_diff == j :
                pred_time_diff = self.df[f'{model}_pred']
                mld_time_diff = self.df[self.target]
            mod_params = getattr(self, f'{model}'.lower())
            for val in mod_params.params.keys() :
                res_values[val][0].append(mod_params.params[val])
                res_values[val][1].append(mod_params.pvalues[val])
            mae.append(np.nanmean(abs(self.df.loc[self.test_df.index, f'{model}_pred'] - self.df.loc[self.test_df.index, self.target])))
            r_squared.append(mod_params.rsquared)
        for val in mod_params.params.keys() :
            res_values[val] = np.array(res_values[val])
        if model == 'OLS' :
            self.OLS_results = res_values
            self.OLS_r_squared = np.array(r_squared)
            self.OLS_mae = np.array(mae)
        elif model == 'GLS':
            self.GLS_results = res_values
            self.GLS_r_squared = np.array(r_squared)
            self.GLS_mae = np.array(mae)
        self.df[f'{model}_pred'] = pred_time_diff
        self.df[f'{model}_{self.target}'] = mld_time_diff