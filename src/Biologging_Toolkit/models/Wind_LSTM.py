import numpy as np
import torch
from tqdm import tqdm
from torch import utils, nn
from torch.autograd import Variable
import time
from typing import Union, List
import netCDF4 as nc
from Biologging_Toolkit.utils.acoustic_utils import *
from Biologging_Toolkit.utils.machine_learning_utils import *
from Biologging_Toolkit.config.config_training import *

class WindLSTM() :

	def __init__(self,
			  depid : Union[str, List] = None,
			  path : Union[str, List] = None,
			  acoustic_path : Union[str, List] = None,
			  *,
			  variable : str = 'wind_speed',
			  dataloader = 'preprocessed',  
			  supplementary_data = []
              ) :

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_params = model_params
		self.variable = variable
      
		# Load data
		self.depid = depid
		if isinstance(self.depid, List) :
			assert len(self.depid) == len(acoustic_path) and len(self.depid) == len(path), "Please provide paths for each depid"
		else :
			self.depid = [self.depid]
			acoustic_path = [acoustic_path]
			path = [path]
		self.ref = self.depid[0] if len(self.depid) == 1 else split_parameters['test_depid']
		self.fns = []
		self.indices = []
		self.depids = []
		self.dives = []
		idx = 0
		for dep_path, dep, ac_path in zip(path, self.depid, acoustic_path) :
			ds = nc.Dataset(dep_path)
			dives = np.unique(ds['dives'][:].data)
			for dive in dives :
				if os.path.exists(os.path.join(ac_path, f'acoustic_dive_{int(dive):05d}.npz')):
					self.fns.append(os.path.join(ac_path, f'acoustic_dive_{int(dive):05d}.npz'))
					self.indices.append(idx)
					self.dives.append(dive)
					idx+=1
					self.depids.append(dep)
		self.fns, self.indices, self.depids, self.dives = np.array(self.fns), np.array(self.indices), np.array(self.depids), np.array(dives)
		self.train_split, self.test_split = get_train_test_split(self.fns, self.indices, self.depids, method = split_parameters['method'], test_depid = split_parameters['test_depid'])

		self.num_epochs = hyperparameters['num_epochs']
		self.batch_size = hyperparameters['batch_size']
		self.learning_rate = hyperparameters['learning_rate']
		self.weight_decay = hyperparameters['weight_decay']
		self.model = RNNModel(self.model_params['input_size'],
                              self.model_params['hidden_size'],
                              1,
                              1).to(self.device)

		self.trainloader = utils.data.DataLoader(LoadDives(self.train_split,
                                                              self.variable, 
                                                              supplementary_data), 
                                                     self.batch_size, shuffle = True)
		self.testloader = utils.data.DataLoader(LoadDives(self.test_split,
                                                             self.variable,
                                                             supplementary_data), 
                                                    self.batch_size)
        
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
                
		self.train_loss = []        
		self.train_accuracy = []
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
				outputs = outputs.view(-1, 1)
				labels = labels.view(-1, 1)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

				loss_batch.append(loss.item())
				acc_batch.append(np.mean(abs(outputs.cpu().detach().numpy() - labels.cpu().detach().numpy())))
                
			self.train_accuracy.append(np.mean(acc_batch))
			self.train_loss.append(np.mean(loss_batch))
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
				outputs = outputs.view(-1, 1)
				labels = labels.view(-1, 1)

				loss = self.criterion(outputs, labels)   
                
				outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
				all_preds.extend(outputs)
				all_labels.extend(labels)   
				acc_test.append(np.mean(abs(outputs - labels)))
				loss_test.append(loss.item())
                
		self.test_accuracy.append(np.mean(acc_test))
		self.test_loss.append(np.mean(loss_test))
		self.all_preds, self.all_labels = all_preds, all_labels

		if np.mean(acc_test) <= np.min(self.test_accuracy) : 
			torch.save(self.model.state_dict(), f"{self.ref}_{self.model_params['input_size']}")
			np.savez(f'best_training_{self.ref}', train_loss = self.train_loss, test_loss = self.test_loss, accuracy = self.test_accuracy, preds = self.all_preds, labels = self.all_labels, train = self.train_split[0], test = self.test_split[0])  
		np.savez('current_epoch', train_loss = self.train_loss, test_loss = self.test_loss, accuracy = self.test_accuracy, preds = self.all_preds, labels = self.all_labels, train = self.train_split[0], test = self.test_split[0])  



class LoadDives(utils.data.Dataset) :

	seq_length = 800

	def __init__(self, split, variable, supplementary_data) :
		self.variable = variable
		self.fns = split[0]
		self.other_inputs = supplementary_data
        
		# Filter out indices with NaN labels
		valid_indices = []
		for idx in range(len(self.fns)):
			data = np.load(self.fns[idx])
			if (data['len_spectro'] < self.seq_length / 15) or (data['len_spectro'] > self.seq_length) :
				continue
			if np.isnan(data[self.variable][-1]) :
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
		spectro = (spectro - normalization['acoustic_min'])/(normalization['acoustic_max'] - normalization['acoustic_min'])
		#Clipping to get rid of outlier values
		spectro[spectro < 0] = 0
		spectro[spectro > 1] = 1

		for other_input in self.other_inputs :
			_data = data[other_input]
			if other_input == 'elevation_angle':
				_data = (_data + np.pi/2) / np.pi
			elif (other_input == 'bank_angle') or (other_input == 'azimuth') :
				_data = (_data + np.pi) / (2*np.pi)
			elif other_input == 'depth' :
				_data = _data / 1300
			spectro = torch.cat((spectro, torch.Tensor(_data).unsqueeze(1)), dim = 1)
            
		# Remove depth under 10m
		spectro = spectro[data['depth'] >= 10] 
        
		if len(spectro) < self.seq_length:
			spectro = torch.cat((torch.zeros(self.seq_length-len(spectro), spectro.size(1)), spectro))

		_label = data[self.variable][-1]
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
		#self.batch_norm = nn.BatchNorm1d(input_dim)
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
