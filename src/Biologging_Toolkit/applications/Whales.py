import os
import pickle
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from typing import Union, List
import netCDF4 as nc
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from Biologging_Toolkit.utils.whale_utils import successive_detections, sliding_window_sum

class Whales():
    """
    A class to handle cetacean annotations.

    Attributes:
        depid: The deployment ID of the animal being analyzed.
        analysis_length: The length of the dive portion (in seconds) that is analyzed to be considered as a drift dive.
    """
    idx = 1

    def __init__(self,
                 depid : Union[List, str],
                 *,
                 annotation_path : Union[List, str] = [''],
                 ):
        """
        Initializes the DriftDives object with the given deployment ID, dataset path, and analysis length.
        Args:
            depid: The deployment ID of the animals being analyzed.
            annotation_path: The path to the annotation data
        """
        self.depid = depid
        self.annotation_path = annotation_path
        if isinstance(self.depid, List):
            assert len(self.depid) == len(self.annotation_path), "Please provide paths for each depid"
        else:
            self.depid = [self.depid]
            self.annotation_path = [self.annotation_path]
        self.annotations = {_dep : pd.read_csv(_annot) for _dep,_annot in zip(self.depid, self.annotation_path)}

        ### Keep only drift dives longer than 3mn
        for key in self.annotations.keys():
            self.annotations[key]['duration'] = self.annotations[key]['end_drift'] - self.annotations[key]['start_drift']
            self.annotations[key] = self.annotations[key][self.annotations[key].duration > 180].reset_index()

    def categorize(self, save = False, save_path = '.'):
        '''
        Adds column baleen, delphinid, spermwhale and other to annotation dataframe
        Dataframe still iterated through drift dives
        '''
        labels = {'other' : ['10 Hz sound', 'Boat', 'Fish', 'Flow noise', 'Grattage', 'Ind 150Hz', 'Ind 150Hz + 300 Hz', 'Tir sismique',
                             'Ind 150Hz + 40 Hz', 'Ind 350Hz', 'Ind 40 Hz', 'Ind 400', 'Ind 40Hz', 'Ind 60 Hz', 'Ind clicks 18kHz',
                             'Unidentified clicks', 'Unknown', 'Unknown clicks', 'Unknown double pulse'],
                  'baleen' : ['ABW', 'Downsweep', 'FW', 'HW', 'MW', 'SRW', 'SW', 'Sweep'],
                  'delphinid' : ['Buzz', 'Clicks', 'Delphinid clicks', 'Delphinid whistle'],
                  'spermwhale' : ['Spermwhale']}
        for key in self.annotations.keys() :
            _other, _baleen, _delphinid, _spermwhale = np.zeros(len(self.annotations[key])),np.zeros(len(self.annotations[key])),np.zeros(len(self.annotations[key])),np.zeros(len(self.annotations[key]))
            for i, row in self.annotations[key].iterrows() :
                if np.any(np.isin(row[['Annotation','Annotation2','Annotation3']], labels['other']) & (row[['Indice','Indice2','Indice3']] >= self.idx)) :
                    _other[i] = 1
                if np.any(np.isin(row[['Annotation','Annotation2','Annotation3']], labels['baleen']) & (row[['Indice','Indice2','Indice3']] >= self.idx)) :
                    _baleen[i] = 1
                if np.any(np.isin(row[['Annotation','Annotation2','Annotation3']], labels['delphinid']) & (row[['Indice','Indice2','Indice3']] >= self.idx)) :
                    _delphinid[i] = 1
                if np.any(np.isin(row[['Annotation','Annotation2','Annotation3']], labels['spermwhale']) & (row[['Indice','Indice2','Indice3']] >= self.idx)) :
                    _spermwhale[i] = 1
            self.annotations[key]['other'] = _other
            self.annotations[key]['baleen'] = _baleen
            self.annotations[key]['delphinid'] = _delphinid
            self.annotations[key]['spermwhale'] = _spermwhale
        if save :
            with open(os.path.join(save_path, f'annotations.pkl'), 'wb') as f:
                pickle.dump(self.annotations, f)

    def day_pooling(self, save = False, save_path = '.') :
        '''
        Create daily pooled dataframe by averaging all data and summing positive minutes and drift dive durations
        '''
        self.daily = {}
        for key in self.annotations.keys() :
            self.annotations[key]['date'] = pd.to_datetime(self.annotations[key]['start_drift'], unit = 's').dt.date
            ### Switch dataframe to date format by pooling data
            self.daily[key] = self.annotations[key][['date','duration', 'lat','lon','other','baleen','delphinid','spermwhale']].groupby('date').agg({'duration':'sum','lat':'mean','lon':'mean','other':'mean','baleen':'mean','delphinid':'mean','spermwhale':'mean'})
            ### Get positive minutes
            _temp = self.annotations[key][['date','duration']][self.annotations[key]['baleen'] == 1].groupby('date').agg('sum')
            self.daily[key] = self.daily[key].join(_temp, on='date', rsuffix='_baleen').fillna({'duration_baleen': 0})
            _temp = self.annotations[key][['date','duration']][self.annotations[key]['delphinid'] == 1].groupby('date').agg('sum')
            self.daily[key] = self.daily[key].join(_temp, on='date', rsuffix='_delphinid').fillna({'duration_delphinid': 0})
            _temp = self.annotations[key][['date','duration']][self.annotations[key]['spermwhale'] == 1].groupby('date').agg('sum')
            self.daily[key] = self.daily[key].join(_temp, on='date', rsuffix='_spermwhale').fillna({'duration_spermwhale': 0})
        if save :
            with open(os.path.join(save_path, f'daily_pool.pkl'), 'wb') as f:
                pickle.dump(self.daily, f)

    def join_auxiliary(self, paths : Union[List, str], pos = True, save = False, save_path = '.'):
        '''
        Join PrCA and Flash data from nc file to daily pooled annotations
        '''
        if isinstance(paths, str):
            paths = [paths] * len(self.depid)
        if pos :
            self.pos = {}
        for path, depid in zip(paths, self.depid) :
            ### Count flash and jerks per dive from nc file
            ds = nc.Dataset(path)
            aux = pd.DataFrame()
            dives = ds['dives'][:].data
            depth = ds['depth'][:].data
            aux['dives'] = np.unique(dives)
            aux['time'] = np.array([ds['time'][:].data[dives == _dive][0].item() for _dive in aux.dives])
            aux['jerk'] = np.array([np.nansum(ds['jerk'][:].data[dives == _dive] >= 400) for _dive in aux.dives])
            aux['bathy'] = np.array([np.nanmean(ds['bathymetry'][:].data[dives == _dive]) for _dive in aux.dives])
            aux['flash'] = np.array([np.nansum(ds['flash'][:].data[:,2][dives == _dive]) for _dive in aux.dives])
            aux['temp'] = np.array([np.nanmean(ds['temperature'][:].data[dives == _dive]) for _dive in aux.dives])
            aux['surface_temp'] = np.array([np.nanmean(ds['temperature'][:].data[(dives == _dive) & (depth <= 10)]) for _dive in aux.dives])
            aux['sal'] = np.array([np.nanmean(ds['salinity'][:].data[dives == _dive]) for _dive in aux.dives])
            aux['surface_sal'] = np.array([np.nanmean(ds['salinity'][:].data[(dives == _dive) & (depth <= 10)]) for _dive in aux.dives])
            aux['date'] = pd.to_datetime(aux['time'], unit='s').dt.date
            aux = aux.groupby('date').agg('mean').reset_index()
            self.daily[depid] = pd.merge(self.daily[depid], aux[['date','jerk','flash', 'temp', 'bathy','sal','surface_temp','surface_sal']], on='date', how='left')
            if pos :
                self.pos[depid] = pd.DataFrame({'lat':ds['lat'][:].data[::100], 'lon':ds['lon'][:].data[::100]})
        if save :
            with open(os.path.join(save_path, f'daily_pool.pkl'), 'wb') as f:
                pickle.dump(self.daily, f)
    def get_pos(self, paths : Union[List, str]):
        if isinstance(paths, str):
            paths = [paths] * len(self.depid)
        self.pos = {}
        for path, depid in zip(paths, self.depid) :
            ds = nc.Dataset(path)
            self.pos[depid] = pd.DataFrame({'lat': ds['lat'][:].data[::100], 'lon': ds['lon'][:].data[::100]})

    def join_CTD(self, paths : Union[List, str], save = False, save_path = '.'):
        for path, depid in zip(paths, self.depid) :
            try :
                ctd_ds = nc.Dataset(path)
            except FileNotFoundError :
                continue
            ctd_time = np.array(
                [(datetime(1950, 1, 1, 0, 0, 0) + timedelta(elem)).replace(tzinfo=timezone.utc).timestamp() for elem in
                 ctd_ds['JULD'][:].data])
            if np.all(ctd_ds['TEMP_ADJUSTED'][:].mask):
                temp_var = 'TEMP'
            else:
                temp_var = 'TEMP_ADJUSTED'
            if np.all(ctd_ds['PSAL_ADJUSTED'][:].mask):
                sal_var = 'PSAL'
            else :
                sal_var = 'PSAL_ADJUSTED'
            if ('CHLA_ADJUSTED' in ctd_ds.variables.keys()) and (np.all(ctd_ds['CHLA_ADJUSTED'][:].mask)) :
                chla_var = 'CHLA'
            else :
                chla_var = 'CHLA_ADJUSTED'
            #chla_var, temp_var, sal_var = 'CHLA', 'TEMP', 'PSAL'
            temp = ctd_ds[temp_var][:].data
            temp[ctd_ds[temp_var][:].mask] = np.nan
            sal = ctd_ds[sal_var][:].data
            sal[ctd_ds[sal_var][:].mask] = np.nan
            if 'CHLA_ADJUSTED' in ctd_ds.variables.keys() :
                chla = ctd_ds[chla_var][:].data
                chla[ctd_ds[chla_var][:].mask] = np.nan
            else :
                chla = np.full(temp.shape, np.nan)
            sal[sal > 100] = np.nan
            temp[temp > 100] = np.nan
            chla[chla > 100] = np.nan
            temp_ctd, temp_surface = [], []
            sal_ctd, sal_surface = [], []
            chla_ctd, chla_surface = [], []
            for temp_profile, sal_profile, chla_profile in zip(temp, sal, chla) :
                #    try:
                temp_ctd.append(np.nanmean(temp_profile))
                temp_surface.append(temp_profile[10])
                sal_ctd.append(np.nanmean(sal_profile))
                sal_surface.append(sal_profile[10])
                chla_ctd.append(np.nanmean(chla_profile))
                chla_surface.append(chla_profile[10])
            #    except ValueError:
            #        temp_ctd.append(np.nan)
            #        temp_surface.append(np.nan)
            #        sal_ctd.append(np.nan)
            #        sal_surface.append(np.nan)
            #        chla_ctd.append(np.nan)
            #        chla_surface.append(np.nan)
            temp_df = pd.DataFrame({'time':ctd_time, 'temp_ctd':temp_ctd, 'temp_surface':temp_surface,
                                    'sal_ctd':sal_ctd, 'sal_surface':sal_surface, 'chla_ctd':chla_ctd, 'chla_surface':chla_surface})
            temp_df['date'] = pd.to_datetime(temp_df['time'], unit='s').dt.date
            temp_df = temp_df.groupby('date').agg('mean').reset_index()
            self.daily[depid] = pd.merge(self.daily[depid], temp_df[['date', 'temp_ctd', 'temp_surface', 'sal_ctd', 'sal_surface', 'chla_ctd', 'chla_surface']],
                                         on='date', how='left', suffixes = ("_old", None))
            #self.daily[depid] = self.daily[depid][[c for c in self.daily[depid].columns if not c.endswith("_old")]]
            #_temp = self.daily[depid]['temp'].to_numpy()
            #_sal = self.daily[depid]['sal'].to_numpy()
            #_surf_temp = self.daily[depid]['surface_temp'].to_numpy()
            #_surf_sal = self.daily[depid]['surface_sal'].to_numpy()
            #temp_ctd = self.daily[depid]['temp_ctd'].to_numpy()
            #sal_ctd = self.daily[depid]['sal_ctd'].to_numpy()
            #surf_temp_ctd = self.daily[depid]['temp_surface'].to_numpy()
            #surf_sal_ctd = self.daily[depid]['sal_surface'].to_numpy()
            #try :
            #    _temp[~np.isnan(temp_ctd)] = temp_ctd[~np.isnan(temp_ctd)]
            #    _sal[~np.isnan(sal_ctd)] = sal_ctd[~np.isnan(sal_ctd)]
            #    _surf_temp[~np.isnan(surf_temp_ctd)] = surf_temp_ctd[~np.isnan(surf_temp_ctd)]
            #    _surf_sal[~np.isnan(surf_sal_ctd)] = surf_sal_ctd[~np.isnan(surf_sal_ctd)]
            #except :
            #    pass
            #self.daily[depid]['temperature'] = _temp
            #self.daily[depid]['salinity'] = _sal
            #self.daily[depid]['surface_temperature'] = _surf_temp
            #self.daily[depid]['surface_salinity'] = _surf_sal
        if save :
            with open(os.path.join(save_path, f'daily_pool.pkl'), 'wb') as f:
                pickle.dump(self.daily, f)

    def get_map_annotation(self):
        df = pd.DataFrame()
        for key in self.annotations:
            df = pd.concat((df, self.annotations[key]), ignore_index=True)
        df.lat = np.round(df.lat * 4) / 4
        df.lon = np.round(df.lon * 4) / 4
        df['year'] = df.date.apply(lambda x: x.year)
        total_duration = df.groupby(['lat', 'lon', 'year'])['duration'].sum().reset_index(name='total_duration')
        df = df[['Annotation', 'Annotation2', 'Annotation3', 'lat', 'lon', 'year', 'duration']].melt(
            id_vars=['lat', 'lon', 'year', 'duration']).dropna()
        df.value[np.isin(df.value, ['Buzz', 'Clicks', 'Delphinid clicks', 'Delphinid whistle'])] = 'Delphinid'
        species = ['ABW', 'FW', 'HW', 'MW', 'SRW', 'SW', 'Delphinid', 'Spermwhale']
        df = df[np.isin(df.value.to_numpy(), species)].drop('variable', axis=1)
        value_duration = (df.dropna(subset=['value'])
                          .groupby(['lat', 'lon', 'year', 'value'])['duration']
                          .sum()
                          .reset_index(name='value_duration'))
        result = value_duration.merge(total_duration, on=['lat', 'lon', 'year'])
        result['proportion'] = result['value_duration'] / result['total_duration']
        self.map_annotation = result

    def simple_logistic_regression(self, depids, ind_var = 'jerk'):
        """
        Logistic regression to predict cetacean presence based on three classes : spermwhale, delphinid and baleen whales
        Split is done using stratified K folds not on target but on independant variables

        Parameters
        ----------
        depids : list of depids to keep from daily dataframe
        ind_var : independant variables / column names to base regression on
        """
        X = pd.DataFrame(pd.concat((self.daily[dep][ind_var] for dep in depids)).reset_index(drop = True))
        y = pd.concat((self.daily[dep][['baleen', 'spermwhale', 'delphinid']] for dep in depids)).reset_index(drop = True)
        combined = pd.concat([X, y], axis=1)
        combined = combined.dropna().reset_index(drop = True)
        X = combined[X.columns]
        y = combined[y.columns]
        y[y > 0] = 1
        skf = StratifiedKFold(n_splits=4)
        y_pred = y.copy()
        for j, _class in enumerate(['baleen','spermwhale','delphinid']):
            for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(X)), y[_class])):
                X_train, X_test = X.loc[train_index], X.loc[test_index]
                y_train, y_test = y.loc[train_index], y.loc[test_index]
                if np.count_nonzero(y_train[_class]) == 0:
                    continue
                logreg = LogisticRegression(class_weight='balanced').fit(X_train, y_train[_class])
                pred = logreg.predict(X_test)
                y_pred.loc[test_index, _class] = pred
        try : self.balreg = LogisticRegression(class_weight='balanced').fit(X, y['baleen'])
        except : print('Baleen whale error')
        try : self.spermreg = LogisticRegression(class_weight='balanced').fit(X, y['spermwhale'])
        except : print('Spermwhale error')
        try : self.delreg = LogisticRegression(class_weight='balanced').fit(X, y['delphinid'])
        except : print('Delphinid error')
        return y, y_pred, X

    def logistic_regression(self, depids, ind_var = ['jerk','flash'], ref = 'label'):
        """
        Logistic regression to predict cetacean presence based on three classes : spermwhale, delphinid and baleen whales
        Split is done using stratified K folds not on target but on independant variables

        Parameters
        ----------
        depids : list of depids to keep from daily dataframe
        ind_var : independant variables / column names to base regression on
        """
        X = pd.concat((self.daily[dep][ind_var] for dep in depids)).reset_index(drop = True)
        y = pd.concat((self.daily[dep][['baleen', 'delphinid', 'spermwhale']] for dep in depids)).reset_index(drop = True)
        y[y > 0] = 1
        skf = StratifiedKFold(n_splits=4)
        y_pred, y_label, X_label = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if ref == 'label' :
            skf_ref = y
        else :
            skf_ref = X.mean(axis = 1).astype(int) if isinstance(ind_var, list) else X.astype(int)
        for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(X)), skf_ref)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            balreg = LogisticRegression(class_weight='balanced').fit(X_train, y_train['baleen'])
            balpred = balreg.predict(X_test)
            spermreg = LogisticRegression(class_weight='balanced').fit(X_train, y_train['spermwhale'])
            spermpred = spermreg.predict(X_test)
            delreg = LogisticRegression(class_weight='balanced').fit(X_train, y_train['delphinid'])
            delpred = delreg.predict(X_test)
            y_label = pd.concat((y_label, y_test))
            X_label = pd.concat((X_label, X_test))
            y_pred = pd.concat((y_pred, pd.DataFrame({'baleen':balpred, 'spermwhale':spermpred, 'delphinid':delpred})))
        self.balreg = LogisticRegression(class_weight='balanced').fit(X, y['baleen'])
        self.spermreg = LogisticRegression(class_weight='balanced').fit(X, y['spermwhale'])
        self.delreg = LogisticRegression(class_weight='balanced').fit(X, y['delphinid'])
        return y_label, y_pred, X_label

    def random_forest(self, depids, ind_var = ['jerk','flash'], ref = 'label'):
        X = pd.concat((self.daily[dep][ind_var] for dep in depids)).reset_index(drop=True)
        y = pd.concat((self.daily[dep][['baleen', 'delphinid', 'spermwhale']] for dep in depids)).reset_index(drop=True)
        y[y > 0] = 1
        y_pred, y_label, X_label = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in range(4):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
            rf = RandomForestClassifier(n_estimators=500, random_state=0)
            rf.fit(X_train, y_train['baleen'])
            balpred = rf.predict(X_test)
            result = permutation_importance(rf, X_test, y_test['baleen'], n_repeats=10, random_state=0)
            perm_importance = pd.Series(result.importances_mean, index=X.columns)
            print(perm_importance.sort_values(ascending=False))
            rf = RandomForestClassifier(n_estimators=500, random_state=0)
            rf.fit(X_train, y_train['spermwhale'])
            result = permutation_importance(rf, X_test, y_test['spermwhale'], n_repeats=10, random_state=0)
            spermpred = rf.predict(X_test)
            perm_importance = pd.Series(result.importances_mean, index=X.columns)
            print(perm_importance.sort_values(ascending=False))
            rf = RandomForestClassifier(n_estimators=500, random_state=0)
            rf.fit(X_train, y_train['delphinid'])
            delpred = rf.predict(X_test)
            result = permutation_importance(rf, X_test, y_test['delphinid'], n_repeats=10, random_state=0)
            perm_importance = pd.Series(result.importances_mean, index=X.columns)
            print(perm_importance.sort_values(ascending=False))
            y_label = pd.concat((y_label, y_test))
            X_label = pd.concat((X_label, X_test))
            y_pred = pd.concat(
                (y_pred, pd.DataFrame({'baleen': balpred, 'spermwhale': spermpred, 'delphinid': delpred})))
        return y_label, y_pred, X_label

    def get_successive_detections(self):
        for key in self.annotations.keys():
            self.annotations[key] = successive_detections(self.annotations[key])

    def get_window_sum(self):
        for key in self.annotations.keys():
            self.annotations[key] = sliding_window_sum(self.annotations[key])

    def load_data(self, annotation_path = None, daily_path = None):
        if annotation_path is not None :
            with open(annotation_path, "rb") as input_file:
                self.annotations = pickle.load(input_file)
        if daily_path is not None :
            with open(daily_path, "rb") as input_file:
                self.daily = pickle.load(input_file)