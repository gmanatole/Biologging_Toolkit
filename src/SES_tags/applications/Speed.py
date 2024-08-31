#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:47:53 2024

@author: grosmaan
"""

import pandas as pd
import seaborn as sns
from angle_orientation import Orientation
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

def pairplot_noise_speed(ml, freq = 100):
	aux = pd.read_csv(f'/run/media/grosmaan/LaCie/individus_brut/individus/{ml}/aux_data.csv')
	spl = pd.read_csv(f'/run/media/grosmaan/LaCie/individus_brut/individus/{ml}/spl_data_cal.csv')
	inst = Orientation(ml)
	lower_bound, upper_bound = aux.time, aux.time+10
	inst.compute_angles()
	low_ind = np.array([np.argmin(abs(inst.t_moy - lb)) for lb in lower_bound])
	upp_ind = np.array([np.argmin(abs(inst.t_moy - ub)) for ub in upper_bound])
	depth_start = inst.P_moy[low_ind]
	depth_end = inst.P_moy[upp_ind]
	dP = depth_end - depth_start
	phi = np.array([np.nanmean(inst.elevation_angle[lb:ub]) for lb,ub in zip(low_ind, upp_ind)])
	fig, ax = plt.subplots(1,2)
	ax[0].scatter(abs(dP/10*np.sin(phi)), spl['2200'], c = dP, s = 2, alpha = 0.5)
	ax[1].scatter(abs(dP/10*np.sin(phi)), aux['SPL_corr'], c = dP, s = 2, alpha = 0.5)
	

def noise_speed(ml, freq = 60):
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
	
	return inertial, _freq

'''	g = sns.jointplot(x="speed", y="spl_db", data=inertial,
	                  kind="kde", truncate=False,
	                color="m", height=7, fill = True)
	sns.regplot(x='speed', y='spl_db', data=inertial, ax=g.ax_joint, scatter=False, order=2, color='r')'''