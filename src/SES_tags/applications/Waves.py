import os
from glob import glob
import numpy as np
from scipy.signal import welch
import netCDF4 as nc
from datetime import datetime, timezone
import soundfile as sf
from SES_tags.wrapper import Wrapper
from SES_tags.utils.format_utils import *



class Waves(Wrapper) :
	
	def __init__(
		self,
		depid, 
		*,
		path,
		sens_path
		) :
		
		super().__init__(
			depid,
			path
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
	
		#Make P the same length as jerk
		self.P = np.pad(self.P[:len(self.jerk)], (0, max(0, len(self.jerk) - len(self.P))), constant_values=np.nan)
	
	def high_resolution_peaks(self, raw_path) :
		"""
		Verify low-res
		olution jerk detections using high-resolution data.
		
		This method reads high-resolution data from the raw sensor files (swv), aligns them with the
		corresponding low-resolution peaks, and detects peaks at a higher sampling rate. The detected
		peaks are stored in the `hr_peaks` attribute.
		
		Parameters
		----------
		raw_path : str
			Path to the directory containing the raw sensor data files (swv).
		
		Notes
		-----
		This method assumes that raw data files are available and that the timestamps are correctly 
		aligned with the low-resolution data. Detected peaks are validated against the high-resolution data.
		"""
		swv_fns = np.array(glob(os.path.join(raw_path, '*swv')))
		xml_fns = np.array(glob(os.path.join(raw_path, '*xml')))
		xml_fns = xml_fns[xml_fns != glob(os.path.join(raw_path, '*dat.xml'))]
		xml_start_time = get_start_date_xml(xml_fns)
		idx_names = get_xml_columns(xml_fns[0], cal='acc', qualifier2='d4') 
		surface_periods = self.get_surface(self.P, 2)
		for i in np.unique(surface_periods) :
			surface = surface_periods == i
			surface_time = self.sens_time[surface]
			fn_idx = np.argmax(xml_start_time[xml_start_time - surface_time[0] < 0])
			fs = sf.info(swv_fns[fn_idx]).samplerate
			sig, fs = sf.read(swv_fns[fn_idx], 
					 start = int((self.sens_time[0] + surface_time[0] - xml_start_time[fn_idx] - 1)*fs),
					 stop = int((self.sens_time[0] + surface_time[-1] - xml_start_time[fn_idx] + 1)*fs))
			A_surf = sig[:, idx_names]
			A_surf = (A_surf * self.A_cal_poly[0] + self.A_cal_poly[1]) @ self.A_cal_map
			
	@staticmethod
	def get_surface(P, threshold) :
		surface = P < threshold
		t = np.linspace(0, len(surface)-1, len(surface))
		t_surface = t[surface]
		count = 0
		surf = []
		for i in range(len(t_surface) -1) :
		    if t_surface[i+1] - t_surface[i] < 60*5 :
		        surf.append(count) 
		    else :
		        count +=1
		        surf.append(count)	
		surf.append(count)
		return np.array(surf)
	
	@staticmethod
	def compute_psd(acc_data, sampling_rate):
	    # Remove mean to detrend the data
	    acc_data = acc_data - np.nanmean(acc_data)
	
	    # Compute PSD using Welch's method
	    freqs, psd = welch(acc_data, fs=sampling_rate, window='hann', nperseg=1024, scaling='density')
	
	    return freqs, psd