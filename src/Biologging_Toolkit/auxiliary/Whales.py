import os
import pickle
import pandas as pd
import numpy as np
from typing import Union, List
import netCDF4 as nc
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
                if np.any(np.isin(row[['Annotation','Annotation2','Annotation3']], labels['other']) & (row[['Indice','Indice2','Indice3']] > self.idx)) :
                    _other[i] = 1
                if np.any(np.isin(row[['Annotation','Annotation2','Annotation3']], labels['baleen']) & (row[['Indice','Indice2','Indice3']] > self.idx)) :
                    _baleen[i] = 1
                if np.any(np.isin(row[['Annotation','Annotation2','Annotation3']], labels['delphinid']) & (row[['Indice','Indice2','Indice3']] > self.idx)) :
                    _delphinid[i] = 1
                if np.any(np.isin(row[['Annotation','Annotation2','Annotation3']], labels['spermwhale']) & (row[['Indice','Indice2','Indice3']] > self.idx)) :
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
            aux['dives'] = np.unique(dives)
            aux['time'] = np.array([ds['time'][:].data[dives == _dive][0].item() for _dive in aux.dives])
            aux['jerk'] = np.array([np.nansum(ds['jerk'][:].data[dives == _dive] >= 400) for _dive in aux.dives])
            aux['flash'] = np.array([np.nansum(ds['flash'][:].data[:,2][dives == _dive]) for _dive in aux.dives])
            aux['date'] = pd.to_datetime(aux['time'], unit='s').dt.date
            aux = aux.groupby('date').agg('mean').reset_index()
            self.daily[depid] = pd.merge(self.daily[depid], aux[['date','jerk','flash']], on='date', how='left')
            self.pos[depid] = pd.DataFrame({'lat':ds['lat'][:].data[::100], 'lon':ds['lon'][:].data[::100]})
        if save :
            with open(os.path.join(save_path, f'daily_pool.pkl'), 'wb') as f:
                pickle.dump(self.daily, f)

    def get_successive_detections(self):
        for key in self.annotations.keys():
            self.annotations[key] = successive_detections(self.annotations[key])

    def get_window_sum(self):
        for key in self.annotations.keys():
            self.annotations[key] = sliding_window_sum(self.annotations[key])