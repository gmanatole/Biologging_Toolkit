import torch
from torch import nn
import numpy as np
import os

def get_train_test_split(paths, indices, depids, method = 'random_split', test_depid = None, split = 0.8) :
	if method == 'depid' :
		if isinstance(test_depid, str) :
			test_depid = [test_depid]
		training = np.where(~np.isin(depids, test_depid))[0]
		testing = np.where(np.isin(depids, test_depid))[0]
		return (paths[training], indices[training]), (paths[testing], indices[testing])

	elif method == 'temporal_split' :
		return (paths[:int(split * len(indices))], indices[:int(split * len(indices))]), (paths[int(split * len(indices)):], indices[int(split * len(indices)):])

	elif method == 'random_split' :
		np.random.seed(32)
		suffle_idx = list(range(len(indices)))
		random.shuffle(suffle_idx)
		indices = [indices[i] for i in suffle_idx]
		paths = [paths[i] for i in suffle_idx]
		return (paths[:int(split * len(indices))], indices[:int(split * len(indices))]), (paths[int(split * len(indices)):], indices[int(split * len(indices)):])
	elif method == 'skf':
		raise NotImplementedError("This method is to be implemented later.")


class WeightedMSELoss(nn.Module):
    '''
    Weighted MSE loss for precipitation data.
    Can be adapted for any unbalanced regression task.
    '''
    def __init__(self, zero_weight=0.1, positive_weight=1.0):
        super(WeightedMSELoss, self).__init__()
        self.zero_weight = zero_weight
        self.positive_weight = positive_weight

    def forward(self, prediction, target):
        weights = torch.where(target == 0, self.zero_weight, self.positive_weight)
        loss = weights * (prediction - target) ** 2
        return torch.mean(loss)

class Loss_LSTM(nn.Module):

    def __init__(self, criterion = 'MSE', variables : [str, list] = 'wind_speed'):
        super(Loss_LSTM, self).__init__()
        self.criterion = criterion
        self.variables = variables
        if isinstance(self.variables, str):  
            self.variables = [self.variables]
            
    def forward(self, prediction, target):
        if len(self.variables) == 1 and self.variables[0] == 'total_precipitation' :
            loss = WeightedMSELoss(zero_weight=0.1, positive_weight=1.0)
            return loss(prediction, target)
        if len(self.variables) == 1 and self.variables[0] == 'wind_speed' :
            loss = nn.MSELoss()
            return loss(prediction, target)
        if len(self.variables) == 2 :
            loss1 = nn.MSELoss()
            loss2 = WeightedMSELoss(zero_weight=0.1, positive_weight=1.0)
            return torch.sum(torch.stack([loss2(prediction[:,i], target[:,i]) 
                               if var == 'total_precipitation' else loss1(prediction[:,i], target[:,i]) 
                              for i, var in enumerate(self.variables)]))

#Normalization still required as well as filtering out above depth 10m if wanted
def preprocess_dive_files(depid, ds, path, fns, variable, supplementary_data, seq_length = 1500) :
    """
    Processes dive files for a given deployment ID and generates `.npz` files containing 
    spectrogram data and supplementary data for each dive.

    This function performs several key preprocessing steps:
    - Extracts spectrogram data based on dive indices from time-synced files.
    - Normalizes spectrogram data, fills in missing data, and applies a depth filter (optional).
    - Includes additional scalar data (specified by the `variable` parameter) as well as supplementary data.

    Each generated `.npz` file contains:
    - `spectro`: 2D array with spectrogram data of shape `(seq_length, num_features)`, padded if necessary.
    - `label`: Dictionary mapping each variable to a scalar or sequence data.
    - Supplementary data from `supplementary_data` as additional arrays.

    Parameters
    ----------
    depid : str
        The deployment ID associated with the dive files.
    ds : xarray.Dataset
        An xarray Dataset containing time series data, depth information, and labels.
    path : str
        Path where the processed `.npz` files will be saved.
    fns : list of str
        List of file paths to the raw spectrogram `.npz` files.
    variable : str or list of str
        Variable name(s) in `ds` to be used as labels. If a single string is provided, it will be converted to a list.
    supplementary_data : list of str
        List of variable names from `ds` to include as supplementary data in the output files.
    seq_length : int, optional
        The desired sequence length for the spectrogram data in the output. Defaults to 1500.

    Returns
    -------
    None
        The function saves processed `.npz` files for each dive, but does not return any data.

    Notes
    -----
    - Spectrogram data from multiple files per dive will be concatenated if needed.
    - Data points with NaN values in `variable` are filtered out.
    - Depth filter is applied to restrict data below a certain depth threshold (10m by default).

    Raises
    ------
    FileNotFoundError
        If any of the specified spectrogram files are not found.
    ValueError
        If the `seq_length` is shorter than the actual length of any spectrogram sequence.
    """
    dives = ds['dives'][:].data[fns != None]
    if isinstance(variable, str):   # If input is a single string
        variable = [variable]
    label = {var : ds[var][:].data[fns != None] for var in variable}
    depth = ds['depth'][:].data[fns != None]
    time_array = ds['time'][:].data[fns != None]        

    other_inputs = {}
    for data in supplementary_data :
        other_inputs[data] = ds[data][:].data[fns != None]               
    fns = fns[fns != None]   

    for idx in np.unique(dives) :
        # Filter out indices with NaN labels
        mask = dives == idx
        _label = {key : label[key][mask] for key in label.keys()}
        if np.any(np.isnan([_label[key][-1] for key in _label.keys()])) or np.any([len(_label[key]) == 0 for key in _label.keys()]):  # Check if the label is not NaN
            continue
        if len(np.unique(fns[mask])) == 1:
            _data = np.load(fns[mask][0])
            pos = np.searchsorted(_data['time'][:len(_data['time'])], time_array[mask])
            spectro = _data['spectro'][pos]
        else :
            spectro = []
            for fn in np.unique(fns[mask]) :
                _mask = mask & (fns == fn)
                _data = np.load(fn)
                pos = np.searchsorted(_data['time'], time_array[_mask])
                spectro.extend(_data['spectro'][pos])

        _other_inputs = {key : other_inputs[key][mask] for key in other_inputs.keys()}
        other_data = {**_other_inputs, **_label}
        spectro = np.array(spectro)
        '''if len(spectro) < seq_length:
            spectro = np.concatenate((np.zeros((seq_length-len(spectro), spectro.shape[1])), spectro))'''
        np.savez(os.path.join(path, f'{depid}_dive_{int(idx):05d}.npz'), 
                 time = time_array[mask],
                 freq = _data['freq'],
                 spectro = np.nan_to_num(spectro),
                 **other_data)
