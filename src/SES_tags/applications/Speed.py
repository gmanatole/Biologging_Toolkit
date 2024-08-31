import pandas as pd
import seaborn as sns
from angle_orientation import Orientation
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from SES_tags.wrapper import Wrapper

class Speed(Wrapper) :

	def __init__(self):
		pass

	def from_inertial(ml, freq = 60):
		timestamps = pd.read_csv(f'/run/media/grosmaan/LaCie/individus_brut/individus/{ml}/welch_timestamps.csv')
		inertial = pd.read_csv(f'/run/media/grosmaan/LaCie/individus_brut/individus/{ml}/inertial_data.csv')
		spl_time, spl = [],[]
		for file in timestamps.fn:
			columns = pd.read_csv(file, nrows=0).columns.to_numpy()[:-1]
			_freq = columns[np.argmin(abs(columns.astype(float)-freq))]
			_temp = pd.read_csv(file, usecols = ['time', _freq])
			spl_time.extend(_temp['time']) ; spl.extend(_temp[_freq])
		dt = spl_time[1] - spl_time[0]; N=int(inertial.epoch.iloc[1]-inertial.epoch.iloc[0])
		
		spl_mean = np.convolve(spl, np.ones(3)/3, mode='valid')
		spl_time_mean = np.convolve(spl_time, np.ones(3)/3, mode='valid')
		_spl = interp1d(spl_time_mean, spl_mean, bounds_error=False)(inertial.epoch)
		inertial[_freq] = np.log10(_spl)
		
		angle = abs(inertial.elevation_angle.to_numpy())
		angle[angle < 0.35] = np.nan

		depth = inertial.depth.to_numpy()
		dP = abs(depth[1:] - depth[:-1])
		dP[dP < 0.5] = np.nan
		
		speed = dP / np.sin(angle[:-1]) / N
		inertial['speed'] = np.append(speed, speed[-1])
		

