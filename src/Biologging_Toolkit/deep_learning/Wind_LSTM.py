import numpy as np
import torch
from tqdm import tqdm
from torch import utils, nn
from torch.autograd import Variable
import time
import netCDF4 as nc
from Biologging_Toolkit.utils.acoustic_utils import *

class WindLSTM() :
	
	def __init__(self,
			  depid : str = None,
			  path : str = None,
			  acoustic_path : str = None,
			  variable : str = 'era',
			  *,
			  num_epochs = 25,
			  learning_rate = 0.001,
			  weight_decay = 0.000,
			  criterion : str = 'MSE'
			  ) :
		
		self.depid = depid
		self.acoustic_path = acoustic_path
		self.ds = nc.Dataset(path)
		self.fns = find_npz_for_time(self.ds['time'][:], self.acoustic_path)
		self.variable = variable

		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.model = RNNModel(513, 512, 1, 1)
		self.dataloader = utils.data.DataLoader(LoadData(self.variable, self.ds, self.fns))
		if criterion == 'MSE':
			self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay = 0.000)
		self.num_epochs = num_epochs

	def train(self):
	
		for epoch in range(self.num_epochs):
			acc_batch = []
			for batch in tqdm(self.dataloader):
	
				self.optimizer.zero_grad()
	
				data, labels = batch
				outputs = self.model(data)
				loss = self.criterion(outputs.squeeze(), labels)
	
				loss.backward()
				self.optimizer.step()
                
				outputs, labels = outputs.squeeze().detach().numpy(), labels.numpy()
				acc_batch.append(np.sum(abs(outputs-labels) < 1)/len(labels))
                
			accuracy.append(np.mean(acc_batch))

			self.test(epoch)
			self.model.train()

			torch.save(model.state_dict(), 'current_training')
			print('model saved')
	
	def test(self, epoch): 
		self.model.eval()
		acc_test, loss_test = [], []
		all_preds, all_labels = [], []
		total_time = 0
		with torch.no_grad():
			for batch in tqdm(self.test_dataloader):
				data, labels = batch
				outputs = self.model(data)
				loss_test.append(self.criterion(outputs.squeeze(), labels))
				outputs, labels = outputs.squeeze().detach().numpy(), labels.numpy()
				all_preds.extend(outputs)
				all_labels.extend(labels)
				acc_test.append(np.sum(abs(outputs-labels) < 1)/len(labels))	
		np.save('results/'+str(epoch+1), np.vstack((all_preds, all_labels)))

class LoadData(utils.data.Dataset) :
	
	seq_length = 1500
	
	def __init__(self, variable, ds, fns):
		self.fns = fns
		self.ds = ds
		self.dives = self.ds['dives'][:].data
		self.variable = variable 
		self.label = self.ds[variable][:].data

	def __len__(self):
		'''
		Returns the number of dives in dataset
		'''
		return len(np.unique(self.dives))
	
	def __getitem__(self, idx):
		mask = (self.dives == idx) & (self.ds['depth'][:].data > 10) 
		label = self.label[mask][-1]
		
		if len(np.unique(self.fns[mask]) == 1):
			_data = np.load(self.fns[mask][0])
			pos = np.searchsorted(_data['time'], self.ds['time'][mask].data)
			spectro = _data['spectro'][pos]
			spectro = torch.Tensor(spectro)
		else :
			spectro = []
			for fn in self.fns[mask] :
				_data = np.load(fn)
				pos = np.unique(np.searchsorted(_data['time'], self.ds['time'][mask].data))
				spectro.extend(_data['spectro'][pos])
			spectro = torch.Tensor(spectro)
		
		if len(spectro) == self.seq_length:
			return spectro, torch.tensor(label, dtype=torch.float)
		else:
			spectro = torch.cat((torch.zeros(self.seq_length-len(spectro), spectro.size(1)), spectro))
			return spectro, torch.tensor(label, dtype=torch.float)


class RNNModel(nn.Module):
	'''
	RNN module that can implement a LSTM
	Number of hiddend im and layer dim are parameters of the model
	'''
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(RNNModel, self).__init__()

		# Number of hidden dimensions
		self.hidden_dim = hidden_dim

		# Number of hidden layers
		self.layer_dim = layer_dim

		# RNN
		#self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
		self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

		self.fc1 = nn.Linear(hidden_dim, 128)

		# Readout layer
		self.fc = nn.Linear(128, output_dim)

	def forward(self, x):

		# Initialize hidden state with zeros
		h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
		c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
		#pdb.set_trace()
		out, (hn, cn) = self.rnn(x, (h0, c0))
		out = self.fc1(out[:, -1, :])
		out = self.fc(out)

		return out
