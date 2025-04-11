import torch
from torch import nn, utils
import numpy as np
from sklearn.neighbors import KernelDensity
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
        #weights = torch.where(target == 0, self.zero_weight, self.positive_weight)
        #weights = torch.tanh(target/100)**2
        weights = 1 - 10 / (torch.pi * 2 * 3) * torch.exp(-(target - 10)**2/(2*3**2))
        #weights[weights < 0.1] = 0.1
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

class MLDDataLoader(utils.data.Dataset):
    def __init__(self, X, Y, ninputs, model = 'MLP'):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.ninputs = ninputs
        self.model = model

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx] + torch.normal(0, 0.07, size=(self.X[idx].size(dim=0),1)).squeeze()
        Y = self.Y[idx] + torch.randint(-10, 10, (1,1)).item()
        shift = torch.randint(0, 5, (1,1)).item()
        X = X[self.generate_shift_indices(X.size(dim=0), shift)]
        if self.model == 'CNN' or 'CNN_LSTM':
            X = torch.FloatTensor(X).reshape(1, self.ninputs, -1)
        return X, Y

    def generate_shift_indices(self, N, shift):
        size_of_data = N // self.ninputs
        mask = [True] * N
        for i in range(self.ninputs):
            mask[i * size_of_data: i * size_of_data + (5 - shift)] = [False]*(5-shift)
            mask[(i+1) * size_of_data - shift: (i+1) * size_of_data] = [False]*shift
        return mask

class RegressionDataAugmenter:
    def __init__(self, X, Y, bandwidth=0.5):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.bandwidth = bandwidth  # Controls smoothness in KDE

    def augment_data(self, target_size=None):
        """
        Oversample underrepresented samples in a regression dataset based on density estimation.

        :param target_size: (Optional) Desired total dataset size after augmentation. If None, doubles the dataset.
        :return: Augmented dataset (X_aug, Y_aug)
        """
        if target_size is None:
            target_size = int(len(self.X) * 1.5)
        kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        kde.fit(self.Y.reshape(-1, 1))
        log_density = kde.score_samples(self.Y.reshape(-1, 1))
        density = np.exp(log_density)
        sample_probs = 1 / (density + 1e-6)
        sample_probs /= sample_probs.sum()
        num_samples_needed = target_size - len(self.X)
        augmented_indices = np.random.choice(len(self.X), size=num_samples_needed, p=sample_probs, replace=True)
        X_aug = np.vstack((self.X, self.X[augmented_indices]))
        Y_aug = np.hstack((self.Y, self.Y[augmented_indices]))
        return X_aug, Y_aug


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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 256], output_size=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[2], output_size)
        )
    def forward(self, x):
        return self.layers(x)

class CNN(nn.Module):
    def __init__(self, input_channels=1, num_filters=16, kernel_size=4, output_size=1, ninputs = 4, size = 40):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, ninputs, size)
            dummy_output = self.conv_layers(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, input_channels=1, num_filters=16, kernel_size=4, output_size=1, ninputs=4, size=40,
                 hidden_dim=128, lstm_layers=1):
        super(CNN_LSTM, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters, kernel_size=kernel_size, stride=1,
                      padding=2),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, ninputs, size)  # Adjusted input shape
            dummy_output = self.conv_layers(dummy_input)
            conv_output_shape = dummy_output.shape  # (batch, channels, height, width)
            flattened_size = conv_output_shape[1] * conv_output_shape[2]  # num_filters * height
        self.lstm = nn.LSTM(input_size=flattened_size, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_layers(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, x.shape[1], -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc_layers(x)
        return x
