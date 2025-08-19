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
			  loss = 'normal',
			  output_dim = 1,
			  model_type = 'LSTM',
			  supplementary_data = [],
			  data_augmentation = False,
			  test_depid = None,
			  fine_tune = False,
			  model_path = None
			  ) :

		### CLASS ATTRIBUTES
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_params = model_params
		self.variable = variable
		self.loss = loss
		self.output_dim = output_dim
		self.data_augmentation = data_augmentation
		if not test_depid :
			test_depid = split_parameters['test_depid']
			
		### LOAD DATA AND CORRECT DIVES
		self.depid = depid
		if isinstance(self.depid, List) :
			assert len(self.depid) == len(acoustic_path) and len(self.depid) == len(path), "Please provide paths for each depid"
		else :
			self.depid = [self.depid]
			acoustic_path = [acoustic_path]
			path = [path]
		self.ref = self.depid[0] if len(self.depid) == 1 else test_depid
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

		### CREATE TRAIN TEST SPLIT
		self.train_split, self.test_split = get_train_test_split(self.fns, self.indices, self.depids, method = split_parameters['method'], test_depid = test_depid)
		self.train_split = np.array(self.train_split)
		self.test_split = np.array(self.test_split)


		### CREATE/LOAD MODEL
		self.num_epochs = hyperparameters['num_epochs']
		self.batch_size = hyperparameters['batch_size']
		self.learning_rate = hyperparameters['learning_rate']
		self.weight_decay = hyperparameters['weight_decay']
		if model_type == 'LSTM':
			self.model = RNNModel(input_dim = self.model_params['input_size'],
                              hidden_dim = self.model_params['hidden_size'],
							  layer_dim = 1,
                              output_dim = self.output_dim).to(self.device)
		if model_type == 'Fusion':
			self.model = FusionLSTM(input_dim = self.model_params['input_size'],
                              hidden_dim = self.model_params['hidden_size'],
                              layer_dim = 1, 
                              output_dim = self.output_dim,
                              acoustic_dim=513,
                              acoustic_out=16).to(self.device)
		if self.data_augmentation : 
			self.train_split = long_tail_rain_augmenter(self.train_split)


		### BUILD DATALOADERS
		self.trainloader = utils.data.DataLoader(LoadDives(self.depid, self.train_split,
                                                              self.variable, 
                                                              supplementary_data,
														      self.data_augmentation,
														      loss = self.loss), 
                                                     self.batch_size, shuffle = True)
		if (test_depid in ['ml17_280a', 'ml18_294b', 'ml18_296a']) and (variable == 'cfosat') :
			variable = 'wind_speed'
		else :
			variable = self.variable
		self.testloader = utils.data.DataLoader(LoadDives(test_depid, self.test_split,
                                                             variable,
                                                             supplementary_data,
														     loss = self.loss), 
                                                    self.batch_size)

		### CREATE LOSS FUNCTION AND OPTIMIZER
		if self.loss == 'weighted': 
			self.criterion = WeightedMSELoss()
		elif self.loss == 'rain' :
			self.criterion = WeightedMSELossRain()
		elif self.loss == 'classes': 
			self.criterion = nn.CrossEntropyLoss()
		else :
			self.criterion = nn.MSELoss()
			
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)

		### OPTIONAL FINE TUNING
		if fine_tune == True :
			self.model.load_state_dict(torch.load(model_path, weights_only=True))
			for name, layer in self.model.named_children():
				if name == 'fc3' :
					for param in layer.parameters():
						param.requires_grad = False
			self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.learning_rate, weight_decay = self.weight_decay)

		### ACCURACY ARRAYS
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
				if self.loss == 'classes' :
					labels = rain_categories(labels)
				data = data.to(self.device)
				labels = labels.to(self.device)
				outputs = self.model(data)
				if self.loss == 'classes' :
					loss = self.criterion(outputs, labels)
				else :
					outputs = outputs.view(-1, 1)
					labels = labels.view(-1, 1)
					loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

				loss_batch.append(loss.item())
				if self.loss == 'classes' :
					probs = torch.softmax(outputs, dim=1)
					predicted_class = torch.argmax(probs, dim=1)
					acc_batch.append((predicted_class == labels).sum().item() / len(labels))
				else :
					acc_batch.append(np.mean(abs(outputs.cpu().detach().numpy() - labels.cpu().detach().numpy())))
                
			self.train_accuracy.append(np.mean(acc_batch))
			self.train_loss.append(np.mean(loss_batch))
			torch.save(self.model.state_dict(), f"{epoch}_{self.ref}_{self.model_params['input_size']}")
			self.test(epoch)
			self.model.train()


	def test(self, epoch): 
		self.model.eval()
		acc_test, loss_test = [], []
		all_preds, all_labels = [], []
		with torch.no_grad():
			for batch in tqdm(self.testloader):
                
				data, labels = batch
				if self.loss == 'classes' :
					labels = rain_categories(labels)
				data = data.to(self.device)
				labels = labels.to(self.device)
                
				outputs = self.model(data)

				if self.loss == 'classes' :
					loss = self.criterion(outputs, labels)
				else :
					outputs = outputs.view(-1, 1)
					labels = labels.view(-1, 1)
					loss = self.criterion(outputs, labels)


				if self.loss == 'classes' :
					probs = torch.softmax(outputs, dim=1)
					predicted_class = torch.argmax(probs, dim=1)
					acc_test.append((predicted_class == labels).sum().item() / len(labels))
					all_preds.extend(predicted_class.cpu().detach().numpy())
					all_labels.extend(labels.cpu().detach().numpy())
				else :
					outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
					acc_test.append(np.mean(abs(outputs - labels)))
					all_preds.extend(outputs)
					all_labels.extend(labels)
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

	def __init__(self, depid, split, variable, supplementary_data, data_augmentation = False, loss = 'normal') :
		self.variable = variable
		self.fns = split[0]
		self.other_inputs = supplementary_data
		self.data_augmentation = data_augmentation
		self.loss = loss
	
		# Filter out indices with NaN labels
		valid_indices = []
		for idx in range(len(self.fns)):
			data = np.load(self.fns[idx])
			#if depid in ['ml17_280a', 'ml18_294b', 'ml18_296a'] :
			#	valid_indices.append(idx)
			#	continue
			if self.variable == 'cfosat' :
				if 1 not in data['cfosat_quality'] :
					continue
				if (data[self.variable] >= 0).sum() <= 0 :
					continue
				if np.isnan(np.nanmean(data['cfosat'][(data['cfosat'] > 0) & (data['cfosat_quality'] == 1)])) :
					continue
			if np.isnan(np.nanmean(data[self.variable])) :
				continue
			if (data['len_spectro'] < self.seq_length / 15) or (data['len_spectro'] > self.seq_length) :
				continue
			if np.isnan(data[self.variable]).all() :
				continue  
			#if data[self.variable][-1] <= 0 :
			#	continue
			valid_indices.append(idx)
		self.indices = np.array(valid_indices)  # Update self.indices with only valid ones

	def __len__(self):
		'''
		Returns the number of dives in dataset
		'''
		return len(self.indices)

	def __getitem__(self, idx):

		data = np.load(self.fns[self.indices[idx]])
		spectro = torch.tensor(data['spectro'], dtype=torch.float32)
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
			elif other_input == 'bathymetry' :
				_data = - _data / 6000
				_data[_data < 0] = 0
			elif other_input == 'distance_to_coast' :
				_data = _data / 1300
			elif other_input == 'wind_speed' :
				_data = _data / 22
			spectro = torch.cat((spectro, torch.tensor(_data, dtype=torch.float32).unsqueeze(1)), dim = 1)
            
		# Remove depth under 10m
		spectro = spectro[data['depth'] >= 10] 
		if self.data_augmentation :
			spectro = DataAugmentation(spectro)()
			
		if len(spectro) < self.seq_length:
			spectro = torch.cat((torch.zeros(self.seq_length-len(spectro), spectro.size(1)), spectro))

		_label = data[self.variable]
		if self.variable == 'cfosat':
			_label = np.nanmean(_label[(_label > 0) & (data['cfosat_quality'] == 1)])
		else :
			_label = np.nanmean(_label)
		#_label = data[self.variable][-1]
		return torch.nan_to_num(spectro), torch.tensor(_label, dtype=torch.float)


class RNNModel(nn.Module):
	'''
	RNN module that can implement a LSTM
	Number of hidden im and layer dim are parameters of the model
	'''
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(RNNModel, self).__init__()

		self.hidden_dim = hidden_dim
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

class FusionLSTM(nn.Module):
    '''
    LSTM model with separate 1D CNN processing for:
    - Main sensors
    - 4 auxiliary sensors
    - 1 prioritized auxiliary sensor
    '''
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, acoustic_dim=513, acoustic_out=16):
        super(FusionLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.acoustic_dim = acoustic_dim
        self.input_dim = input_dim 
        self.acoustic_out = acoustic_out
		
        # CNN for acoustic data
        self.main_cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.acoustic_dim, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
			nn.Conv1d(64, self.acoustic_out, kernel_size=3, padding=1),
			nn.ReLU())

        # Final fused feature
        final_size = self.input_dim-self.acoustic_dim+self.acoustic_out
        self.rnn = nn.LSTM(input_size=final_size, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        '''
        x: (batch, seq_len, 500) — 495 main + 5 auxiliary (last is prioritized)
        '''
        spec = x[:, :, :self.acoustic_dim]
        aux = x[:, :, self.acoustic_dim:]
        
        spec = spec.permute(0, 2, 1) 
        spec = self.main_cnn(spec)
        spec = spec.permute(0, 2, 1)
        fused = torch.cat([spec, aux], dim=2)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(fused, (h0, c0))
        out = out[:, -1, :]

        out = self.act1(self.fc1(out))
        out = self.act2(self.fc2(out))
        out = self.fc3(out)

        return out
