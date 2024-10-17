import os
from glob import glob
import numpy as np
from scipy.signal import welch, find_peaks, butter, filtfilt, sosfilt
import netCDF4 as nc
from datetime import datetime, timezone
import soundfile as sf
from Biologging_Toolkit.processing.Dives import Dives
from Biologging_Toolkit.utils.format_utils import *



class Waves(Dives) :
	'''
	(Hashimoto and Konbune, 1988)
	Earle (1996, p. 12) 
	From : do elephant seals act as weather buoys ?
	'''
	
	def __init__(
		self,
		depid, 
		*,
		path,
		sens_path : str = None,
		raw_path : str = None,
		data = {'time': None, 'A' : None}
		) :
		
		super().__init__(
			depid,
			path = path
        )
		
		if sens_path :
			data = nc.Dataset(sens_path)
			self.samplerate = data['P'].sampling_rate
			self.P = data['P'][:].data
			self.A = data['A'][:].data
			length = np.max([self.A.shape[1], len(self.P)])
			self.sens_time = datetime.strptime(data.dephist_deploy_datetime_start, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() + np.arange(0, length/self.samplerate, np.round(1/self.samplerate,2))
			self.A_cal_poly = data['A'].cal_poly[:].reshape(2, 3)
			self.A_cal_map = data['A'].cal_map[:].reshape(3, 3)
		elif data['A'] is not None and data['time'] is not None :
			self.A = data['A']
			self.inertial_time = data['time']
			self.samplerate = self.inertial_time[1]-self.inertial_time[0]
	
		self.raw_path = raw_path
		
		
	def get_wave_period(self, samplerate = 200) :
		"""
		
		This method reads high-resolution data from the raw sensor files (swv), aligns them with the
		corresponding low-resolution peaks, and aims to compute wave_period. 
		
		Parameters
		----------
		raw_path : str
			Path to the directory containing the raw sensor data files (swv).
		"""
		elevation_angle, time_angle = self.ds['elevation_angle'][:].data, self.ds['time'][:].data
		swv_fns = np.array(glob(os.path.join(self.raw_path, '*swv')))
		xml_fns = np.array(glob(os.path.join(self.raw_path, '*xml')))
		xml_fns = xml_fns[xml_fns != glob(os.path.join(self.raw_path, '*dat.xml'))].flatten()
		xml_start_time = get_start_date_xml(xml_fns)
		idx_names = np.array(get_xml_columns(xml_fns[0], cal='acc', qualifier2='d4'))
		if samplerate == 50 :
			idx_names = idx_names[:,0]
			
		period, period_time, med_period, zc_period = [], [], [], []
		
		#Get parameters for pass band filter
		params = butter(N=4, fs = samplerate, Wn=[1/12, 1/5], btype='bandpass', analog=False, output = 'sos')

		surface_mask, surface_periods = self.get_surface(self.P, 2)
		for i in np.unique(surface_periods) :
			surface = surface_periods == i
			surface_time = self.sens_time[surface_mask][surface]
			fn_idx = np.argmax(xml_start_time[xml_start_time - surface_time[0] < 0])
			fs = sf.info(swv_fns[fn_idx]).samplerate
			sig, fs = sf.read(swv_fns[fn_idx], 
					 start = int((surface_time[0] - xml_start_time[fn_idx] - 1)*fs),
					 stop = int((surface_time[-1] - xml_start_time[fn_idx] + 1)*fs))
			A_surf = np.column_stack([sig[:,idx_names[i]].flatten() for i in range(len(idx_names))])
			A_surf = (A_surf * self.A_cal_poly[0] + self.A_cal_poly[1]) @ self.A_cal_map
			
			ea = np.nanmean(elevation_angle[(time_angle > surface_time[0]) & (time_angle < surface_time[-1])])
			A_filtered = sosfilt(params, A_surf[:,0]*np.sin(ea) + A_surf[:,2]*np.cos(ea))

			#A_filtered = sosfilt(params, A_surf[:,0])
			
			# Find frequency by identifying peaks in Ax (ie inversion in direction)
			Ax_peaks, _ = find_peaks(A_filtered, prominence = 1.5, distance = 200)
			_med_period = np.nanmedian(np.diff(Ax_peaks))/samplerate
			
			# Find frequency using zero crossings
			zero_crossings = np.where(np.diff(np.sign(A_filtered)))[0]
			_zc_period = np.nanmedian(np.diff(zero_crossings))/samplerate
			
			# Find peaks using FFT
			spectrum = np.abs(np.fft.fft(A_filtered))
			freqs = np.fft.fftfreq(n = len(spectrum), d = 1/samplerate)
			psd_peaks, _ = find_peaks(spectrum[freqs > 0], threshold = 0.01)
			try :
				peak = np.argmax(spectrum[freqs > 0][psd_peaks])
				period.append(1 / freqs[freqs > 0][peak])
			except ValueError:
				period.append(np.nan)
			period_time.append(np.mean(surface_time))
			med_period.append(_med_period)
			zc_period.append(_zc_period)
		self.zc_period = np.array(zc_period)
		self.period = np.array(period)
		self.period_time = np.array(period_time)
		self.med_period = np.array(med_period)
		
		
	@staticmethod
	def get_surface(P, threshold) :
		surface = P < threshold
		t = np.linspace(0, len(surface)-1, len(surface))
		t_surface = t[surface]
		count = 0
		surf = [0]
		for i in range(len(t_surface) -1) :
		    if t_surface[i+1] - t_surface[i] < 60*5 :
		        surf.append(count) 
		    else :
		        count +=1
		        surf.append(count)	
		return surface, np.array(surf)
	
	
		
'''
@staticmethod
def compute_psd(acc_data, sampling_rate):
    # Remove mean to detrend the data
    acc_data = acc_data - np.nanmean(acc_data)
    acc_data = np.concatenate((np.zeros(8000), acc_data, np.zeros(8000)))
    nperseg = 8192
    # Compute PSD using Welch's method
    freqs, psd = welch(acc_data, fs=sampling_rate, window='hann', nperseg=nperseg, noverlap=int(nperseg*0.8), scaling='density')

    return freqs, psd
'''