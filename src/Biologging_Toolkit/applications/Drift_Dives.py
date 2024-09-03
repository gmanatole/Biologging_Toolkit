import matplotlib.pyplot as plt
import matplotlib.colors as co
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import umap.umap_ as umap
import hdbscan
import imageio.v3 as iio
from sklearn.manifold import TSNE
import scipy
import netCDF4 as nc
import time, datetime, calendar
from Biologging_Toolkit.wrapper import Wrapper

class DriftDives(Wrapper) :

	def __init__(self) :
		pass

	def from_inertial(self, inertial, len_analysis = 60, method = 'depth'):
		'''
		:param inertial: inertial dataframe with epoch, depth and bank_angle
		:param len_analysis: length in seconds of dive portion that is analyzed to be considered as drift dive, defaults to 60
		:return: DataFrame with dive type column
		'''
		len_analysis = int(len_analysis/3/2)
		dive_type = np.full(len(inertial), 0)
		for j in range(len_analysis, len(inertial)-len_analysis-1) :
			if ((abs(inertial.bank_angle.iloc[j-len_analysis:j+len_analysis].to_numpy()) > 2).sum() / len(inertial.iloc[j-len_analysis:j+len_analysis]) > 0.98):
				dive_type[j] =  1 

	def from_depth(self, depth):
		len_analysis = int(len_analysis/3)
		dive_type = np.full(len(inertial), 0)
		for j in range(len_analysis, len(inertial)-len_analysis-1) :
			vertical_speed_before = (inertial.depth.iloc[j] - inertial.depth.iloc[j-len_analysis])/(inertial.epoch.iloc[j] - inertial.epoch.iloc[j-len_analysis]) 
			vertical_speed_after = (inertial.depth.iloc[j+len_analysis] - inertial.depth.iloc[j])/(inertial.epoch.iloc[j+len_analysis] - inertial.epoch.iloc[j]) 
			if (abs(vertical_speed_after) < 0.4) or abs((vertical_speed_before) < 0.4) :
				 dive_type[j] = 1


	def acoustic_cluster(self, ml, var = '2200', min_cluster_size = 20, min_samples = 15):
		spl_df = pd.read_csv(f'/run/media/grosmaan/LaCie/individus_brut/individus/{ml}/spl_data_cal.csv').drop('time', axis = 1)
		aux_df = pd.read_csv(f'/run/media/grosmaan/LaCie/individus_brut/individus/{ml}/aux_data.csv')
		df = pd.concat((aux_df, spl_df), axis = 1).dropna(subset = var)
		indices = get_indices(df.depth)
		
		X = np.zeros((len(indices)-1, 100))
		data = pd.DataFrame()
		timestamp, depth = np.zeros((len(indices)-1, 100)), np.zeros((len(indices)-1, 100))
		for i in range(1, len(indices)):
			X[i-1] = scipy.signal.resample(df[var].iloc[indices[i-1]:indices[i]], 100)
			timestamp[i-1] = np.linspace(df.time.iloc[indices[i-1]], df.time.iloc[indices[i]], 100)
			depth[i-1] = scipy.signal.resample(df.depth.iloc[indices[i-1]:indices[i]], 100)
		#X = np.lib.stride_tricks.sliding_window_view(df[var], window_shape=(window_size))
		#data = pd.DataFrame()
		#data['depth'] = df.depth[window_size//2 : -window_size//2+1]
		#data['timestamp'] = df.time[window_size//2 : -window_size//2+1]


		project = umap.UMAP()
		embed = project.fit_transform(X)
		
		clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(embed) 
		data['cluster'] = clusterer.labels_
		data.cluster[data.cluster == -1] = -100
		norm = co.Normalize(vmin=clusterer.labels_.min(), vmax=clusterer.labels_.max())
		cmap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
		for i in range(len(indices)-1):
			plt.scatter(timestamp[i], depth[i], color = cmap.to_rgba([int(clusterer.labels_[i])]), s = 2)
		#plt.plot(data.timestamp, data.depth, c = data.cluster)
		#plt.plot(df.time, df[var])
		plt.show()
		return timestamp, depth, clusterer.labels_, embed

	def acoustic_threshold(self):
		pass



	
