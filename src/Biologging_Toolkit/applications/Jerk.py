import numpy as np
from glob import glob
import os
from Biologging_Toolkit.wrapper import Wrapper
import netCDF4 as nc
from datetime import datetime, timezone
from Biologging_Toolkit.utils.inertial_utils import * 
from Biologging_Toolkit.utils.format_utils import *
import soundfile as sf
from scipy.interpolate import interp1d

class Jerk(Wrapper):
	"""
	A class to process and analyze jerk data from sensor measurements.
	Based on Chevallay, 2024, Hunting tactics of southern elephant seals Mirounga leonina and anti-predatory behaviours of their prey.
	https://doi.org/10.3354/meps14582
	
	The `Jerk` class is designed to work with low-resolution (such as 5Hz data from sens file) and high-resolution (such as 50 Hz data from svw file) jerk data.
	It identifies peaks in the jerk signal that are above a specified threshold.
	The code is aimed to detect jerks from low-resolution data before checking them with high-resolution data.
	
	Attributes
	----------
	lr_threshold : float
		Threshold value for detecting peaks in low-resolution jerk data (default is 200).
	lr_blanking : float
		Blanking criterion in seconds for low-resolution jerk data (default is 5 seconds).
	lr_duration : float, optional
		Minimum duration in seconds of peaks in low-resolution jerk data. Defaults to the inverse of the sampling rate.
	hr_threshold : float
		Threshold value for detecting peaks in high-resolution jerk data (default is 400).
	hr_blanking : float
		Blanking criterion in seconds for high-resolution jerk data (default is 0.25 seconds).
	hr_duration : float
		Minimum duration in seconds of peaks in high-resolution jerk data (default is 0.02 seconds).
	samplerate : float
		Sampling rate of the jerk data.
	jerk : np.ndarray
		The jerk data.
	P : np.ndarray
		Pressure data corresponding to the jerk measurements.
	A : np.ndarray
		Accelerometer data used for calculating jerk.
	sens_time : np.ndarray
		Array of POSIX timestamps (UTC) corresponding to the sensor data.
	A_cal_poly : np.ndarray
		Calibration polynomial for the accelerometer data.
	A_cal_map : np.ndarray
		Calibration map for the accelerometer data.
	lr_peaks : dict
		Dictionary containing peak information for low-resolution jerk data.
	hr_peaks : dict
		Dictionary containing peak information for high-resolution jerk data.
	
	Parameters
	----------
	depid : str
		Deployment ID for the sensor data.
	path : str
		Path to the directory containing the dataset file.
	sens_path : str
		Path to the sensor data file.
	data : dict, optional
		If sens_path not provided.
		Dictionary containing preloaded data with keys 'time', 'jerk', and 'P'. Default is {'time': None, 'jerk': None, 'P': None}.
	"""

	lr_threshold = 200
	lr_blanking = 5
	lr_duration = None
	hr_threshold = 400
	hr_blanking = 25
	hr_duration = 0.02
	
	
	def __init__(self, 
			  depid, 
			  *,
			  path,
			  sens_path = None,
			  raw_path = None,
			  data = {'time': None, 'jerk' : None, 'P' : None}
			  ) :
		
		"""
		Initialize the Jerk class with deployment ID, data path, and sensor data.
		
		This constructor loads the jerk, pressure, accelerometer, and magnetometer data 
		from the specified NetCDF file or from a provided dictionary.
		
		Parameters
		----------
		depid : str
			Deployment ID for the sensor data.
		path : str
			Path to the directory containing the data files.
		sens_path : str
			Path to the sensor data NetCDF file.
		data : dict, optional
			Preloaded data with keys 'time', 'jerk', and 'P'. Default is {'time': None, 'jerk': None, 'P': None}.
		
		Notes
		-----
		The data dictionary is used if sens_path is not provided. The 'time', 'jerk', and 'P' keys should 
		correspond to arrays containing the respective data. If sens_path is provided, data is loaded directly 
		from the NetCDF file located at that path.
		"""
		
		super().__init__(
			depid,
			path
        )
		
		if sens_path :
			data = nc.Dataset(sens_path)
			self.samplerate = data['P'].sampling_rate
			self.P = data['P'][:].data
			self.jerk = data['J'][:].data if 'J' in data.variables.keys() else np.full((len(self.P)), np.nan)
			self.A = data['A'][:].data if 'A' in data.variables.keys() else np.full((3, len(self.P)), np.nan)
			self.M = data['M'][:].data if 'M' in data.variables.keys() else np.full((3, len(self.P)), np.nan)
			length = np.max([len(self.jerk), self.A.shape[1], self.M.shape[1], len(self.P)])
			self.sens_time = datetime.strptime(data.dephist_device_datetime_start, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() + np.arange(0, length/self.samplerate, np.round(1/self.samplerate,2))
			self.A_cal_poly = data['A'].cal_poly[:].reshape(2, 3)
			self.A_cal_map = data['A'].cal_map[:].reshape(3, 3)
		
		elif data['time'] is not None and data['jerk'] is not None and data['P'] is not None :
			self.sens_time, self.jerk, self.P = data['time'], data['jerk'], data['P']
			self.samplerate = np.round(1 / (self.sens_time[1]-self.sens_time[0]), 2)

		self.raw_path = raw_path

		#Make P the same length as jerk
		self.P = np.pad(self.P[:len(self.jerk)], (0, max(0, len(self.jerk) - len(self.P))), constant_values=np.nan)
		
		#Remove surface data
		self.jerk[self.P <= 20] = np.nan
		if not self.lr_duration :
			self.lr_duration = 1 / self.samplerate


	def __call__(self, overwrite = False, resolution = 'high') :
		
		"""
		Process and store jerk peaks in reference dataset.
		
		This method identifies and stores jerk peaks either using low or high resolution data.
		It can optionally overwrite existing jerk data in the NetCDF dataset.
		
		Parameters
		----------
		overwrite : bool, optional
			If True, overwrites any existing 'jerk' variable in the NetCDF dataset. Default is False.
		resolution : str, optional
			Specifies the resolution of peaks to process. Can be 'high' or 'low' or 'low_check'. Default is 'high'.
		"""
		
		if resolution == 'high':
			self.high_resolution_peaks()
			peaks = self.hr_peaks
			threshold = self.hr_threshold
			blanking = self.hr_blanking
			duration = self.hr_duration
			np.savez(os.path.join(self.path, f'{self.depid}_hr_jerks'), self.hr_peaks)
		
		elif resolution == 'low_check' :
			self.low_resolution_peaks()
			self.check_peaks()
			peaks = self.hr_peaks
			threshold = self.hr_threshold
			blanking = self.hr_blanking
			duration = self.hr_duration
		else :
			self.low_resolution_peaks()
			peaks = self.lr_peaks
			threshold = self.lr_threshold
			blanking = self.lr_blanking
			duration = self.lr_duration
			
		jerks = np.full(len(self.ds['time']), 0)
		indices = np.searchsorted(self.ds['time'][:].data, peaks['max_time'], side='right') - 1
		jerks[indices] = peaks['max']
		
		if overwrite :
			if 'jerk' in self.ds.variables:
				self.remove_variable('jerk')
				
		if 'jerk' not in self.ds.variables:
			jerk = self.ds.createVariable('jerk', np.float64, ('time',))
			jerk.units = 'm.s**2'
			jerk.long_name = f'{resolution} resolution jerks'
			jerk.raw_data_samplerate = self.raw_samplerate
			jerk.threshold = threshold
			jerk.threshold_units = 'm.s**2'
			jerk.blanking = blanking
			jerk.blanking_units = 's'
			jerk.duration = duration
			jerk.duration_comment = 'Maximum duration of peaks'
			jerk.duration_units = 's'
			jerk[:] = jerks
	
	def low_resolution_peaks(self) :
		"""
		Detect peaks using low-resolution jerk data.
		
		This method processes the jerk data at a low sampling rate to identify peaks that exceed the 
		specified threshold and respect the blanking and duration conditions.
		The detected peaks, along with their properties (start time, end time, max time, etc.), are stored in the `lr_peaks` attribute.
		
		"""
		
		cc, cend, peak_time, peak_max, minlen = self.get_peaks(self.jerk, self.samplerate, self.lr_blanking, self.lr_threshold, self.lr_duration)
		self.lr_peaks = {}
		self.lr_peaks['start_time'] = cc / self.samplerate  #in seconds
		self.lr_peaks['end_time'] = cend / self.samplerate  #in seconds
		self.lr_peaks['max_time'] = peak_time / self.samplerate #in seconds
		self.lr_peaks['max'] = peak_max
		self.lr_peaks['duration'] = self.lr_peaks['end_time'] - self.lr_peaks['start_time']
		self.lr_peaks['depth'] = self.P[cc]
		peak_times = self.sens_time[0] + self.lr_peaks['max_time']
		self.lr_peaks['datetime'] = np.array(list(map(lambda x : datetime.fromtimestamp(x), peak_times)))
		self.lr_peaks['timestamp'] =  peak_times
		self.raw_samplerate = 50


	def high_resolution_peaks(self, samplerate = 200) :
		"""
		Get peaks in jerk data based on high resolution raw data.
		
		This method reads high-resolution data from the raw sensor files (swv) and detects peaks.
		The detected peaks are stored in the `hr_peaks` attribute.
		
		Parameters
		----------
		raw_path : str
			Either 50 or 200. Samplerate used to read DTAG4 accelerometer data.
		
		Notes
		-----
		It is recommanded to used low_resolution_peaks and check_peaks (forward method) to save computational resources.
		"""
		swv_fns = np.array(glob(os.path.join(self.raw_path, '*swv')))
		xml_fns = np.array(glob(os.path.join(self.raw_path, '*xml')))
		try :
			xml_fns = xml_fns[xml_fns != glob(os.path.join(self.raw_path, '*dat.xml'))].flatten()
		except ValueError :
			pass
		xml_start_time = get_start_date_xml(xml_fns)
		self.hr_samplerate = samplerate
		if samplerate == 50 :
			idx_names = idx_names[:,0]
		
		hr_peaks = {'start_time':[],'end_time':[],'max_time':[],'max':[],'duration':[],'depth':[]}	
		idx_names = get_xml_columns(xml_fns[0], cal='acc', qualifier2='d4') 
		
		for i, swv_fn in enumerate(swv_fns) :
			sig, fs = sf.read(swv_fn)
			A = np.column_stack([sig[:,idx_names[i]].flatten() for i in range(len(idx_names))])
			A = (A * self.A_cal_poly[0] + self.A_cal_poly[1]) @ self.A_cal_map
			jerk = self.norm_jerk(A, fs)
			peaks = self.get_peaks(jerk = jerk,
									samplerate = fs,
									threshold = self.hr_threshold,
									blanking = self.hr_blanking,
									duration = self.hr_duration)
			if peaks is not None :
				cc, cend, peak_time, peak_max, minlen = peaks 
				hr_peaks['start_time'].extend(xml_start_time[i] + cc / samplerate)
				hr_peaks['end_time'].extend(xml_start_time[i] + cend / samplerate)
				hr_peaks['max_time'].extend(xml_start_time[i] + peak_time / samplerate)
				hr_peaks['max'].extend(peak_max)
		hr_peaks['start_time'] = np.array(hr_peaks['start_time'])
		hr_peaks['end_time'] = np.array(hr_peaks['end_time'])
		hr_peaks['max_time'] = np.array(hr_peaks['max_time'])
		indices = np.searchsorted(self.ds['time'][:], hr_peaks['max_time'], side='right') - 1
		hr_peaks['duration'] = hr_peaks['end_time'] - hr_peaks['start_time']
		hr_peaks['depth'] = self.ds['depth'][:].data[indices]
		mask = hr_peaks['depth'] >= 20
		hr_peaks = {key: np.array(value)[mask] for key, value in hr_peaks.items()}
		hr_peaks['samplerate'] = samplerate
		self.raw_samplerate = samplerate
		hr_peaks['banking'] = self.hr_blanking
		hr_peaks['threshold'] = self.hr_threshold
		hr_peaks['duration'] = self.hr_duration
		self.hr_peaks = {key: np.array(value) for key, value in hr_peaks.items()}
			
		
	def distance_detection(self, jerk_data = 'high') :
		'''
		jerk_data : str
			Where to get jerk detections from. 
			'high' needs high resolution detection dictionary. Recommanded. 
			'low' requires low resolution detection dictionary.
			'ref' requires jerk data to be saved in reference structure.
		'''
		if jerk_data == 'high' :
			jerk_time = self.hr_peaks['start_time']
		elif jerk_data == 'low' :
			jerk_time = self.lr_peaks['start_time']
		else :
			jerk_time = self.ds['time'][:][self.ds['jerk'][:] > 0] 
		swv_fns = np.array(glob(os.path.join(self.raw_path, '*swv')))
		xml_fns = np.array(glob(os.path.join(self.raw_path, '*xml')))
		xml_fns = xml_fns[xml_fns != glob(os.path.join(self.raw_path, '*dat.xml'))].flatten()
		xml_start_time = get_start_date_xml(xml_fns)
		idx_names = get_xml_columns(xml_fns[0], cal='acc', qualifier2='d4') 
		if self.hr_samplerate == 50 :
			idx_names = idx_names[:,0]
		indices = np.searchsorted(xml_start_time, jerk_time)-1
		
		for i, swv_fn in enumerate(swv_fns) :
			sig, fs = sf.read(swv_fn)
			A = np.column_stack([sig[:,idx_names[i]].flatten() for i in range(len(idx_names))])
			A = (A * self.A_cal_poly[0] + self.A_cal_poly[1]) @ self.A_cal_map
			for jerk in jerk_time[indices == i] :
				_jerk_time = jerk - xml_start_time[i]
				approach = A[int((_jerk_time - 15) * self.hr_samplerate) : int((_jerk_time - 1) * self.hr_samplerate)]
	      
	def check_peaks(self, samplerate = 200) :
		"""
		Verify low-resolution jerk detections using high-resolution data from DTAG4.
		
		This method reads high-resolution data from the raw sensor files (swv), aligns them with the
		corresponding low-resolution peaks, and detects peaks at a higher sampling rate. The detected
		peaks are stored in the `hr_peaks` attribute.	
		
		Parameters
		----------
		raw_path : str
			Either 50 or 200. Samplerate used to read DTAG4 accelerometer data.
		
		Notes
		-----
		This method assumes that raw data files are available and that the timestamps are correctly 
		aligned with the low-resolution data. Detected peaks are validated against the high-resolution data.
		"""
		swv_fns = np.array(glob(os.path.join(self.raw_path, '*swv')))
		xml_fns = np.array(glob(os.path.join(self.raw_path, '*xml')))
		xml_fns = xml_fns[xml_fns != glob(os.path.join(self.raw_path, '*dat.xml'))].flatten()
		xml_start_time = get_start_date_xml(xml_fns)
		if samplerate == 50 :
			idx_names = idx_names[:,0]
		hr_peaks = {'start_time':[],'end_time':[],'max_time':[],'max':[],'duration':[],'depth':[],'datetime':[],'timestamp':[]}	
		idx_names = get_xml_columns(xml_fns[0], cal='acc', qualifier2='d4') 
		for i, peak_time in enumerate(self.lr_peaks['timestamp']) :
			fn_idx = np.argmax(xml_start_time[xml_start_time - peak_time < 0])
			fs = sf.info(swv_fns[fn_idx]).samplerate
			sig, fs = sf.read(swv_fns[fn_idx], 
					 start = int((self.sens_time[0] + self.lr_peaks['start_time'][i] - xml_start_time[fn_idx] - 1)*fs),
					 stop = int((self.sens_time[0] + self.lr_peaks['end_time'][i] - xml_start_time[fn_idx] + 1)*fs))
			A_peak = np.column_stack([sig[:,idx_names[i]].flatten() for i in range(len(idx_names))])
			A_peak = (A_peak * self.A_cal_poly[0] + self.A_cal_poly[1]) @ self.A_cal_map
			jerk_peak = self.norm_jerk(A_peak, fs)
			jerk_validation = self.get_peaks(jerk = jerk_peak,
									samplerate = fs,
									threshold = self.hr_threshold,
									blanking = self.hr_blanking,
									duration = self.hr_duration)
			if jerk_validation is not None:
				hr_peaks['start_time'].append(self.lr_peaks['start_time'][i])
				hr_peaks['end_time'].append(self.lr_peaks['end_time'][i])
				hr_peaks['max_time'].append(self.lr_peaks['max_time'][i])
				hr_peaks['max'].append(self.lr_peaks['max'][i])
				hr_peaks['duration'].append(self.lr_peaks['duration'][i])
				hr_peaks['depth'].append(self.lr_peaks['depth'][i])
				hr_peaks['datetime'].append(self.lr_peaks['datetime'][i])
				hr_peaks['timestamp'].append(self.lr_peaks['timestamp'][i])
		self.hr_peaks = {key: np.array(value) for key, value in hr_peaks.items()}
		self.raw_samplerate = samplerate
		
		
	@staticmethod
	def get_peaks(jerk, samplerate, blanking, threshold, duration = None):
		"""
		Identify peaks in the jerk signal that exceed a specified threshold.
		
		This static method detects the start, end, and maximum of peaks in the jerk signal that
		are above the provided threshold and shorter than provided duration.
		It also applies a blanking criterion to prevent multiple detections of the same peak.
		
		Parameters
		----------
		jerk : np.ndarray
			1D array of jerk data.
		samplerate : float
			Sampling frequency of the jerk signal in Hz.
		blanking : float
			Blanking criterion in seconds. Peaks closer together than this value will be merged.
		threshold : float
			Threshold value above which a peak is considered valid.
		duration : float, optional
			Minimum duration of a peak in seconds. Peaks shorter than this value will be ignored. Default is None.
		
		Returns
		-------
		cc : np.ndarray
			Array of start indices of detected peaks.
		cend : np.ndarray
			Array of end indices of detected peaks.
		peak_time : np.ndarray
			Array of indices for maximum of detected peaks
			Array of times (since device started logging) at which the peak maximum occurs in seconds.
		peak_max : np.ndarray
			Array of maximum values of the detected peaks.
		minlen : float
			Minimum length of the detected peaks in seconds.
		
		Notes
		-----
		The method converts the blanking and duration values from seconds to samples using the 
		provided sampling rate. It also merges peaks that are within the blanking distance.
		"""
		
		#Go from seconds to number of bins
		blanking *= samplerate
		
		#Find jerk peaks above threshold
		dxx = np.diff((jerk >= threshold).astype(int))
		cc = np.where(dxx > 0)[0] + 1
		if len(cc) == 0:
			return None
		
		# Find ending sample of each peak
		coff = np.where(dxx < 0)[0] + 1
		cend = np.full(len(cc), len(jerk))
		for k in range(len(cc)):
			kends = np.where(coff > cc[k])[0]
			if len(kends) > 0:
				cend[k] = coff[kends[0]]
		
		# Eliminate detections which do not meet blanking criterion & merge pulses that are within blanking distance
		done = False
		while not done:
			kg = np.where(cc[1:] - cend[:-1] > blanking)[0]
			done = len(kg) == (len(cc) - 1)
			cc = cc[np.concatenate(([0], kg + 1))]
			cend = cend[np.concatenate((kg, [len(cend) - 1]))]
		if cend[-1] == len(jerk):
			cc = cc[:-1]
			cend = cend[:-1]
		
		# Remove peaks shorter than duration attribute
		if duration :
			duration *= samplerate
			k = np.where(cend - cc >= duration)[0]
			cc = cc[k]
			cend = cend[k]
			minlen = duration / samplerate
		else:
			minlen = 1 / samplerate
		
		# Determine the time and maximum of each peak
		peak_time = np.zeros(len(cc))
		peak_max = np.zeros(len(cc))
		for a in range(len(cc)):
			segment = jerk[cc[a]:cend[a]]
			index = np.argmax(segment)
			peak_time[a] = index + cc[a]
			peak_max[a] = np.max(segment)
		
		return cc, cend, peak_time, peak_max, minlen


	# Function taken from animaltags Python package
	@staticmethod
	def norm_jerk(A, sampling_rate):
		"""
		Compute the norm of the jerk signal from accelerometer data.
		This static method calculates the norm of the jerk signal, which is the magnitude of 
		the derivative of the acceleration. The norm is computed for each time step in the accelerometer data.
		
		Parameters
		----------
		A : np.ndarray or dict
			Array of accelerometer data (Nx3 for N samples in 3 axes) or a dictionary containing the data.
		sampling_rate : float
			Sampling frequency of the accelerometer signal in Hz.
		
		Returns
		-------
		jerk : np.ndarray
			Array of jerk values, with the same length as the input data.
		
		Notes
		-----
		If `A` is a dictionary, it should contain the accelerometer data under the key 'data' 
		and the sampling rate under 'sampling_rate'. The method returns the norm of the jerk signal.
		"""

		if isinstance(A, dict):
			sampling_rate = A["sampling_rate"]
			a = A["data"]
			jerk = A.copy()
			jerk["data"] = np.concatenate((sampling_rate * np.sqrt(np.sum(np.diff(a, axis=0)**2, axis=1)), [0]))
			jerk["creation_date"] = datetime.now().isoformat()
			jerk["type"] = "njerk"
			jerk["full_name"] = "norm jerk"
			jerk["description"] = jerk["full_name"]
			jerk["unit"] = "m/s3"
			jerk["unit_name"] = "meters per seconds cubed"
			jerk["unit_label"] = "m/s^3"
			jerk["column_name"] = "jerk"
		else:
			a = A
			jerk = sampling_rate * np.concatenate((np.sqrt(np.sum(np.diff(a, axis=0)**2, axis=1)), [0]))
		
		return jerk


