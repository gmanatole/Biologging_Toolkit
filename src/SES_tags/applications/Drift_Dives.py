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
from SES_tags.wrapper import Wrapper

class Drift_Dive(Wrapper) :

	def __init__(self) :
		pass

def get_indices(P):
	# GET SURFACE TIMES
	mask = np.array(P < 1.5)
	surface = [-1]
	k = 0
	for i in range(1, len(mask)):
		if mask[i] == False :
			surface.append(-1)
		elif mask[i] == mask[i-1]:
			surface.append(k)
		else :
			k+=1
			surface.append(k)
	surface = np.array(surface)
	indices = np.array([np.mean(np.where((surface == i))) for i in range(0, max(surface)+1)])
	mask = np.full(len(indices), False)
	for i in range(1, len(indices)) :
		if (indices[i]-indices[i-1] >= 5):
			mask[i] = True
	return indices[mask].astype(int)

def find_drift_dives(inertial, len_analysis = 60, method = 'depth'):
	'''
	:param inertial: inertial dataframe with epoch, depth and bank_angle
	:param len_analysis: length in seconds of dive portion that is analyzed to be considered as drift dive, defaults to 60
	:return: DataFrame with dive type column
	'''
	if method == 'bank_angle':
		len_analysis = int(len_analysis/3/2)
		dive_type = np.full(len(inertial), 0)
		for j in range(len_analysis, len(inertial)-len_analysis-1) :
			if ((abs(inertial.bank_angle.iloc[j-len_analysis:j+len_analysis].to_numpy()) > 2).sum() / len(inertial.iloc[j-len_analysis:j+len_analysis]) > 0.98):
				dive_type[j] =  1 #np.nanstd(inertial.bank_angle.iloc[j-len_analysis:j+len_analysis].to_numpy())
	if method == 'depth' : 
		len_analysis = int(len_analysis/3)
		dive_type = np.full(len(inertial), 0)
		for j in range(len_analysis, len(inertial)-len_analysis-1) :
			vertical_speed_before = (inertial.depth.iloc[j] - inertial.depth.iloc[j-len_analysis])/(inertial.epoch.iloc[j] - inertial.epoch.iloc[j-len_analysis]) 
			vertical_speed_after = (inertial.depth.iloc[j+len_analysis] - inertial.depth.iloc[j])/(inertial.epoch.iloc[j+len_analysis] - inertial.epoch.iloc[j]) 
			if (abs(vertical_speed_after) < 0.4) or abs((vertical_speed_before) < 0.4) :
				 dive_type[j] = 1
	'''peaks, _ = scipy.signal.find_peaks(- inertial.depth, height = -2, distance = 120)
	for i, peak in enumerate(peaks[:-1]) :
		if dive_type[peak : peaks[i+1]].sum() > 20 :
			dive_type[peak : peaks[i+1]][dive_type[peak : peaks[i+1]] == 1] = 2'''
	inertial['dive_type'] = dive_type
	return inertial

def cluster_dives(ml, var = '2200', min_cluster_size = 20, min_samples = 15):
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

def threshold_dives(ml):
	sens = nc.Dataset(f'/run/media/grosmaan/LaCie/individus_brut/CTD/{ml}/{ml}sens5.nc')
	depth, depth_dt, depth_start = sens['P'][:].data, np.round(1/sens['P'].sampling_rate, 2), get_start_date(sens.dephist_device_datetime_start)
	time = np.linspace(0, len(depth), len(depth))*depth_dt+depth_start
	plt.plot()
	
	
def check_cluster(cluster, inertial) :
	timestamp, depth, clusters = cluster['timestamp'], cluster['depth'], cluster['cluster']
	for cluster in np.unique(clusters) :
		_timestamp = timestamp[clusters == cluster]
		print(len(_timestamp))
		percentage = np.full(len(_timestamp), 0)
		for i, dive in enumerate(_timestamp) : 
			if inertial.dive_type[(inertial.epoch > dive.min()) & (inertial.epoch < dive.max())].sum() > 0:
				percentage[i] = 1
		print('Cluster ', cluster, ' contains  :', f'{percentage.sum()/len(percentage):.2f}% of dives that are drift dives')
	_inertial = inertial[inertial.dive_type == 1]
	begin, end = timestamp[:,0], timestamp[:,-1]
	dive_cluster = []
	for i, row in _inertial.iterrows():
		for j, (start, stop) in enumerate(zip(begin, end)):
			if start <= row.epoch <= stop :
				dive_cluster.append(clusters[j])
				continue
	dive_cluster = np.array(dive_cluster)
	print([f'{(dive_cluster == cluster).sum()} dives are in cluster {cluster}' for cluster in np.unique(clusters)])
	
