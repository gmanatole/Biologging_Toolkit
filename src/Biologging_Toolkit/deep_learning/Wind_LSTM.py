import numpy as np
import torch
from tqdm import tqdm
from torch import utils
import time
import netCDF4 as nc

class WindLSTM() :
	
	def __init__(self,
			  depid : str = None,
			  path : str = None,
			  acoustic_path : str = None,
			  variable : str = 'era'
			  ) :
		
		self.depid = depid
		self.acoustic_path = acoustic_path
		self.ds = nc.Dataset(path)
		self.dives = self.ds['dives'][:].data
		self.depth = self.ds['depth'][:].data
		self.variable = variable

		self.model = 'RNN'
		self.dataloader = LoadData(self.acoustic_time, self.variable, self.dives, self.depth)
		self.criterion = ''
		self.optimizer = 'Adam'
	
	def train(self, model, dataloader, test_loader, criterion, optimizer, num_epochs, accuracy):
	
		for epoch in range(num_epochs):
			t0 = time.time()
			acc_batch = []
			for batch in tqdm(dataloader):
	
				optimizer.zero_grad()
	
				data, labels = batch
				outputs = model(data)
				loss = criterion(outputs.squeeze(), labels)
	
				loss.backward()
				optimizer.step()
	
				outputs, labels = outputs.squeeze().detach().numpy(), labels.numpy()
				acc_batch.append(np.sum(abs(outputs-labels) < 1)/len(labels))
			accuracy.append(np.mean(acc_batch))
			if (epoch+1) % 1 == 0:
				self.test(model, test_loader, epoch)
				model.train()
			torch.save(model.state_dict(), 'current_training')
			print('model saved')
	
	def test(self, model, dataloader, epoch, criterion): 
		model.eval()
		acc_test, loss_test = [], []
		all_preds, all_labels = [], []
		total_time = 0
		t3 = time.time()
		with torch.no_grad():
			for batch in tqdm(dataloader):
				t0 = time.time()
				data, labels = batch
				outputs = model(data)
				loss_test.append(criterion(outputs.squeeze(), labels))
				outputs, labels = outputs.squeeze().detach().numpy(), labels.numpy()
				all_preds.extend(outputs)
				all_labels.extend(labels)
				acc_test.append(np.sum(abs(outputs-labels) < 1)/len(labels))	
				t1 = time.time()
				total_time = total_time + (t1-t0)
		t2 = time.time()
		np.save('results/'+str(epoch+1), np.vstack((all_preds, all_labels)))
		print('TIME = ', total_time, t3-t2)


class LoadData(utils.data.DataLoader) :
	
	seq_length = 500
	
	def __init__(self, acoustic, variable, dives, depth):
		self.acoustic = acoustic
		self.variable = variable 
		self.dives = dives
		self.depth = depth

	def __len__(self):
		'''
		Returns the number of dives in dataset
		'''
		return len(np.unique(self.dives)[0])
	
	def __getitem__(self, idx):
		#TRY USING X ARRAY LAZY LOADING
		label = self.variable[self.dives == idx][-1]
		spectro = torch.Tensor(self.acoustic[:].data[self.dives == idx])
		
		if len(spectro) == self.seq_length:
			return spectro, torch.tensor(label, dtype=torch.float)
		else:
			spectro = torch.cat((torch.zeros(self.seq_length-len(spectro), spectro.size(1)), spectro))
			return spectro, torch.tensor(label, dtype=torch.float)
