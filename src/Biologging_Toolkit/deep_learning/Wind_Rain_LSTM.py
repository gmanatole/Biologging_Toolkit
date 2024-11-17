import numpy as np
import torch
from tqdm import tqdm
from torch import utils, nn
from torch.autograd import Variable
import time
from typing import Union
import netCDF4 as nc
from Biologging_Toolkit.utils.acoustic_utils import *
from Biologging_Toolkit.utils.machine_learning_utils import *

class WindLSTM() :

	def __init__(self,
			  depid : str = None,
			  path : str = None,
			  acoustic_path : str = None,
			  *,
			  variables : Union[str, list] = 'wind_speed',
			  split = 0.8,
			  batch_size = 32,           
			  num_epochs = 15,
			  learning_rate = 0.0005,
			  weight_decay = 0.0001,
			  criterion : str = 'MSE',
			  dataloader = 'preprocessed',  
			  supplementary_data = [],              
			  **kwargs,
			  ) :

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		default = {'input_size':513, 'hidden_size':1024}
		self.model_params = {**default, **kwargs}
        
		# Load data
		self.depid = depid
		self.acoustic_path = acoustic_path
		self.ds = nc.Dataset(path)
		self.fns = find_npz_for_time(self.ds['time'][:], self.acoustic_path)
		self.variables = variables
		if isinstance(self.variables, str):   # If input is a single string
			self.variables = [self.variables]
		self.split = split

		# Get split
		indices = np.unique(self.ds['dives'][:].data)
		np.random.seed(32)
		np.random.shuffle(indices)
		self.train_indices = indices[:int(self.split * len(indices))]
		self.test_indices = indices[int(self.split * len(indices)):]
        
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.model = RNNModel(self.model_params['input_size'],
                              self.model_params['hidden_size'],
                              1,
                              len(self.variables)).to(self.device)
		if dataloader == 'raw':        
			self.trainloader = utils.data.DataLoader(LoadData(self.ds, 
                                                              self.fns, 
                                                              self.variables, 
                                                              self.train_indices, 
                                                              supplementary_data), 
                                                     self.batch_size)
			self.testloader = utils.data.DataLoader(LoadData(self.ds, 
                                                             self.fns, 
                                                             self.variables, 
                                                             self.test_indices, 
                                                             supplementary_data), 
                                                    self.batch_size)
		if dataloader == 'preprocessed':        
			self.trainloader = utils.data.DataLoader(LoadDives(self.acoustic_path,
                                                              self.train_indices,
                                                              self.variables, 
                                                              supplementary_data), 
                                                     self.batch_size)
			self.testloader = utils.data.DataLoader(LoadDives(self.acoustic_path,
                                                             self.test_indices,
                                                             self.variables,
                                                             supplementary_data), 
                                                    self.batch_size)
        
		'''if criterion == 'MSE':
			self.criterion = nn.MSELoss()
		self.tp_criterion = WeightedMSELoss(zero_weight=0.1, positive_weight=1.0) '''
		self.criterion = Loss_LSTM()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
		self.num_epochs = num_epochs
		self.loss = []        
		self.accuracy = []
		self.test_accuracy = []
		self.test_loss = []

	def train(self):
		for epoch in range(self.num_epochs):
			acc_batch = []
			loss_batch = []
			for batch in tqdm(self.trainloader):

				self.optimizer.zero_grad()

				data, labels = batch
				data = data.to(self.device)
				labels = labels.to(self.device)

				outputs = self.model(data)
				outputs = outputs.view(-1, len(self.variables))
				labels = labels.view(-1, len(self.variables))

				#loss = self.criterion(outputs.squeeze(), labels)
				'''loss = torch.sum(torch.stack([self.tp_criterion(outputs[:,i], labels[:,i]) 
                               if var == 'total_precipitation' else self.criterion(outputs[:,i], labels[:,i]) 
                              for i, var in enumerate(self.variables)]))'''
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

				loss_batch.append(loss.item())
				acc_batch.append([np.mean(abs(outputs[:,i].cpu().detach().numpy() - labels[:,i].cpu().detach().numpy())) for i in range(len(self.variables))])
                
			self.accuracy.append(np.mean(acc_batch))
			self.loss.append(np.mean(loss_batch))
			self.test(epoch)
			self.model.train()


	def test(self, epoch): 
		self.model.eval()
		acc_test, loss_test = [], []
		all_preds, all_labels = [], []
		with torch.no_grad():
			for batch in tqdm(self.testloader):
                
				data, labels = batch
				data = data.to(self.device)
				labels = labels.to(self.device)
                
				outputs = self.model(data)
				outputs = outputs.view(-1, len(self.variables))
				labels = labels.view(-1, len(self.variables))

				#loss_test.append(self.criterion(outputs.squeeze(), labels))
				'''loss_test.append(torch.sum(torch.stack([self.tp_criterion(outputs[:,i], labels[:,i]) 
                               if var == 'total_precipitation' else self.criterion(outputs[:,i], labels[:,i]) 
                              for i, var in enumerate(self.variables)])))'''
				loss = self.criterion(outputs, labels)   
                
				outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
				all_preds.extend(outputs)
				all_labels.extend(labels)   
				acc_test.append([np.mean(abs(outputs[:,i] - labels[:,i])) for i in range(len(self.variables))])
				loss_test.append(loss.item())
                
		self.test_accuracy.append(np.mean(acc_test))
		self.test_loss.append(np.mean(loss_test))
		self.all_preds, self.all_labels = all_preds, all_labels

		if np.mean(acc_test) < np.min(self.test_accuracy) : 
			torch.save(self.model.state_dict(), f"{self.depid}_{len(self.variables)}_{self.model_params['input_size']}")
			np.savez(f'best_training_{self.depid}_{len(self.variables)}', loss = self.test_loss, accuracy = self.test_accuracy, preds = self.all_preds, labels = self.all_labels, train = self.train_indices, test = self.test_indices)  
		np.savez('current_epoch', loss = self.test_loss, accuracy = self.test_accuracy, preds = self.all_preds, labels = self.all_labels, train = self.train_indices, test = self.test_indices)  


class LoadData(utils.data.Dataset) :

	seq_length = 1500

	def __init__(self, ds, fns, variables, indices, supplementary_data) :
		self.variables = variables 
		self.fns = fns
		self.ds = ds
		self.indices = indices
		self.dives = self.ds['dives'][:].data
		self.label = {var : self.ds[var][:].data for var in variables}
		self.depth = self.ds['depth'][:].data
		self.time_array = self.ds['time'][:].data   
        
		self.other_inputs = {}
		for data in supplementary_data :
			self.other_inputs[data] = self.ds[data][:].data         
		self.fns = self.fns
        
		# Filter out indices with NaN labels
		valid_indices = []
		for idx in self.indices:
			mask = (self.dives == idx)
			_label = {key : self.label[key][mask] for key in self.label.keys()}
			if np.any(np.isnan([_label[key][-1] for key in _label.keys()])) :  # Check if the label is not NaN
				continue
			valid_indices.append(idx)
		self.indices = np.array(valid_indices)  # Update self.indices with only valid ones'''

	def __len__(self):
		'''
		Returns the number of dives in dataset
		'''
		return len(self.indices)

	def __getitem__(self, idx):
		idx = self.indices[idx]
		mask = (self.dives == idx) 
		_label = np.array([self.label[key][mask][-1]-self.label[key][mask][0] if key == 'total_precipitation' else self.label[key][mask][-1] for key in self.label.keys()])
		if len(np.unique(self.fns[mask])) == 1:
			_data = np.load(self.fns[mask][0])
			pos = np.searchsorted(_data['time'], self.time_array[mask])
			spectro = _data['spectro'][pos]
			spectro = torch.Tensor(spectro)
		else :
			spectro = []
			for fn in np.unique(self.fns[mask]) :
				_mask = mask & (self.fns == fn)
				_data = np.load(fn)
				pos = np.searchsorted(_data['time'], self.time_array[_mask])
				spectro.extend(_data['spectro'][pos])
			spectro = torch.Tensor(np.array(spectro))
            
		for data in self.other_inputs :
			spectro = torch.cat((spectro, torch.Tensor(self.other_inputs[data][mask]).unsqueeze(1)), dim = 1)
		spectro = (spectro + 78)/(49 +78)

		if len(spectro) < self.seq_length:
			spectro = torch.cat((torch.zeros(self.seq_length-len(spectro), spectro.size(1)), spectro))
		return torch.nan_to_num(spectro), torch.tensor(_label, dtype=torch.float)

class LoadDives(utils.data.Dataset) :

	seq_length = 1500

	def __init__(self, path, indices, variables, supplementary_data) :
		self.variables = variables 
		self.path = path
		self.other_inputs = supplementary_data
		self.fns = np.sort(glob(os.path.join(path, '*')))
		indices = indices[indices < len(self.fns)].astype(int)
		self.fns = self.fns[indices]
        
		# Filter out indices with NaN labels
		valid_indices = []
		for idx in range(len(self.fns)):
			data = np.load(self.fns[idx])
			if (data['len_spectro'] < self.seq_length / 1005) or (data['len_spectro'] > self.seq_length) :
				continue
			if data['wind_speed'][-1] == np.nan :
				continue
			if data['total_precipitation'][-1] == np.nan :
				continue      
			valid_indices.append(idx)
		self.indices = np.array(valid_indices)  # Update self.indices with only valid ones

	def __len__(self):
		'''
		Returns the number of dives in dataset
		'''
		return len(self.indices)

	def __getitem__(self, idx):

		data = np.load(self.fns[self.indices[idx]])
		spectro = torch.Tensor(data['spectro'])

		spectro = (spectro + 78)/(49 +78)
		for other_input in self.other_inputs :
			spectro = torch.cat((spectro, torch.Tensor(data[other_input]).unsqueeze(1)), dim = 1)

		if len(spectro) < self.seq_length:
			spectro = torch.cat((torch.zeros(self.seq_length-len(spectro), spectro.size(1)), spectro))
		_label = np.array([data[key][-1]-data[key][0] if key == 'total_precipitation' else data[key][-1] for key in self.variables])
		return torch.nan_to_num(spectro), torch.tensor(_label, dtype=torch.float)

    
class RNNModel(nn.Module):
	'''
	RNN module that can implement a LSTM
	Number of hidden im and layer dim are parameters of the model
	'''
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(RNNModel, self).__init__()

		# Number of hidden dimensions
		self.hidden_dim = hidden_dim

		# Number of hidden layers
		self.layer_dim = layer_dim

		# RNN
		self.batch_norm = nn.BatchNorm1d(input_dim)
		#self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
		self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
		self.fc1 = nn.Linear(hidden_dim, 512)
		self.act1 = nn.LeakyReLU()
		self.fc2 = nn.Linear(512, 256)
		self.act2 = nn.LeakyReLU()
		self.fc3 = nn.Linear(256, output_dim)
                

	def norm(self, x) :
		x = x.permute(0, 2, 1)
		x = self.batch_norm(x)
		x = x.permute(0, 2, 1)
		print(torch.max(x), torch.min(x))        
		return x

	def forward(self, x):               
		# Initialize hidden state with zeros
		h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(x.device)
		c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(x.device)
		out, (hn, cn) = self.rnn(x, (h0, c0))
		out = self.fc1(out[:, -1, :])
		out = self.act1(out)
		out = self.fc2(out)
		out = self.act2(out)
		out = self.fc3(out)

		return out
