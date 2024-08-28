import numpy as np
from glob import glob
import os
from SES_tags.wrapper import Wrapper
import netCDF4 as nc
from datetime import datetime, timezone
from SES_tags.utils.inertial_utils import * 
from SES_tags.utils.format_utils import *
import soundfile as sf
from scipy.interpolate import interp1d

class Jerk(Wrapper):
	"""
	A class to process and analyze jerk data from sensor measurements.
	
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
	hr_blanking = 0.25
	hr_duration = 0.02
	
	def __init__(self, 
			  depid, 
			  *,
			  path,
			  sens_path,
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
			self.jerk = data['J'][:].data
			self.P = data['P'][:].data
			self.A = data['A'][:].data
			self.M = data['M'][:].data
			length = np.max([len(self.jerk), self.A.shape[1], self.M.shape[1], len(self.P)])
			self.sens_time = datetime.strptime(data.dephist_deploy_datetime_start, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() + np.arange(0, length/self.samplerate, np.round(1/self.samplerate,2))
			self.A_cal_poly = data['A'].cal_poly[:].reshape(2, 3)
			self.A_cal_map = data['A'].cal_map[:].reshape(3, 3)
			self.M_cal_poly = data['M'].cal_poly[:].reshape(-1, 3)
			self.M_cal_cross = data['M'].cal_cross[:].reshape(-1, 3)
			self.M_cal_tseg = data['M'].cal_tseg[:]
			self.M_cal_map = data['M'].cal_map[:].reshape(3, 3)
			
		elif data['time'] is not None and data['jerk'] is not None and data['P'] is not None :
			self.sens_time, self.jerk, self.P = data['time'], data['jerk'], data['P']
			self.samplerate = np.round(1 / (self.sens_time[1]-self.sens_time[0]), 2)
		
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
			Specifies the resolution of peaks to process. Can be 'high' or 'low'. Default is 'high'.
		"""
		
		self.low_resolution_peaks()
		
		if resolution == 'high' :
			self.high_resolution_peaks()
			peaks = self.hr_peaks
			threshold = self.hr_threshold
			blanking = self.hr_blanking
			duraion = self.hr_duration
		else :
			peaks = self.lr_peaks
			threshold = self.lr_threshold
			blanking = self.lr_blanking
			duration = self.lr_duration
			
		jerks = np.full(len(self.ds['time']), 0)
		indices = np.searchsorted(self.ds['time'][:].data, peaks['timestamp'], side='right') - 1
		jerks[indices] = peaks['max']
		
		if overwrite :
			if 'jerk' in self.ds.variables:
				self.remove_variable('jerk')
				
		if 'jerk' not in self.ds.variables:
			jerk = self.ds.createVariable('jerk', np.float64, ('time',))
			jerk.units = 'm.s**2'
			jerk.long_name = 'High resolution jerks'
			jerk.threshold = threshold
			jerk.threshold_units = 'm.s**2'
			jerk.blanking = blanking
			jerk.blanking_units = 's'
			jerk.duration = duration
			jerk.duration_comment = 'Maximum duration of peaks'
			jerk.duration_units = 's'
			jerk[:] = jerks
		

	def high_resolution_peaks(self, raw_path) :
		"""
		Verify low-resolution jerk detections using high-resolution data.
		
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
		hr_peaks = {'start_time':[],'end_time':[],'max_time':[],'max':[],'duration':[],'depth':[],'datetime':[],'timestamp':[]}		
		for i, peak_time in enumerate(self.lr_peaks['timestamp']) :
			fn_idx = np.argmax(xml_start_time[xml_start_time - peak_time < 0])
			fs = sf.info(swv_fns[fn_idx]).samplerate
			sig, fs = sf.read(swv_fns[fn_idx], 
					 start = int((self.sens_time[0] + self.lr_peaks['start_time'][i] - xml_start_time[fn_idx] - 1)*fs),
					 stop = int((self.sens_time[0] + self.lr_peaks['end_time'][i] - xml_start_time[fn_idx] + 1)*fs))
			A_peak = sig[:, :3]
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
		

	@staticmethod
	def get_peaks(jerk, samplerate, blanking, threshold, duration):
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
	
	
	


'''class Jerk:

    def __init__(self, x, x2, fs=16, fc=2.64, timeDays=0):
        self.x = np.array(x) if not isinstance(x, np.ndarray) else x
        self.x2 = np.array(x2) if not isinstance(x2, np.ndarray) else x2
        self.fs = fs
        self.fc = fc
        self.timeDays = timeDays
        self.timeRef = np.copy(self.x[:, 0])
        self.filtered_x = None
        self.kmeans_model = None


	def fullbw_jerks(self):
		if dys is None:
			dys = [0]
		if len(dys) == 1 or dys[1] < 1:
			dys.append(len(self.P) / (self.samplerate * 24 * 3600))
		
		dd = np.array(dys) * 24 * 3600
		JT = {
		'time': [], 'dur': [], 'dpth': [], 'rmsJ': [], 'cond': [], 'angle': [],
		'maxA': [], 'maxJ': [], 'maxAan': [], 'rmsMJ': [], 'maxMJ': []
		}
		
		while True:
			if len(dd) == 2 and dd[0] >= dd[1]:
				break
			print(f'Processing day {dd[0] / (24 * 3600):.1f} of {dys[1]:.1f}')
		
			X = d3getswv(dd[0] + np.array([0, 6 * 3600]), recdir, depid)
			if len(X['x']) == 0:
				break
		
			Ps = crop_to(P_data, pfs, dd[0] + np.array([0, 6 * 3600]))
			MM = crop_to(M_data, pfs, dd[0] + np.array([0, 6 * 3600]))
			
			As = np.vstack(X['x'][:3])
			As = (As * A_cal_poly[:, 0] + A_cal_poly[:, 1]) @ A_cal_map
			
			Ms = np.vstack(X['x'][3:6])
			kk = nearest(M_cal_tseg, dd[0], -1)
			mpol = M_cal_poly[:, 2 * (kk - 1):(2 * kk)]
			mcross = M_cal_cross[:, 3 * (kk - 1):(3 * kk)]
			Ms = (Ms * mpol[:, 0] + mpol[:, 1]) @ mcross
			Ms = decdc(Ms, 2) @ M_cal_map
			mfs = X['fs'][3] / 2
			MJ = self.njerk(Ms, mfs) / np.nanmean(norm(MM))
			
			afs = X['fs'][0]
			Js = self.njerk(self.A, self.samplerate)
			psi = interp2length(Ps, pfs, afs, len(As))
			Js[psi < 20] = np.nan
			jpk = peak_finder(Js, afs, thr, [BLNK, MINDUR], 0)
			
			if len(jpk['start_time']) > 0:
				n = len(jpk['start_time'])
				JT['time'].extend(jpk['start_time'] + dd[0])
				dd[0] += 6 * 3600
			else:
				dd[0] += 6 * 3600
				continue
			
			Af = comp_filt(As, afs, [3, 50])
			Ah = Af[1]
			R = np.full((n, 8), np.nan)
			
			for k in range(n):
				kp = np.arange(round(afs * (jpk['start_time'][k] - 0.15)), round(afs * (jpk['end_time'][k] + 0.15)))
				if np.any(kp < 1) or np.any(kp > len(As)):
					continue
				R[k, 0] = np.sqrt(np.nanmean(np.diff(As[kp, :], axis=0)**2))
				
				km = np.arange(round(mfs * (jpk['start_time'][k] - 0.15)), round(mfs * (jpk['end_time'][k] + 0.15)))
				if np.any(km < 1) or np.any(km > len(MJ)):
					continue
				R[k, 1] = np.sqrt(np.nanmean(MJ[km]**2))
				R[k, 2] = np.max(Js[kp])
				R[k, 7] = np.nanmax(MJ[km])
				
				ah = Ah[kp, :]
				m, km_max = np.max(norm2(ah)), np.argmax(norm2(ah))
				R[k, 3:5] = [m, np.real(np.arccos(np.abs(ah[km_max, 0] / m)))]
				
				if not np.any(np.isnan(ah)):
					V, D = eig(ah.T @ ah)
					D = np.diag(D)
					D_sorted_indices = np.argsort(D)
					R[k, 5:7] = [D[2] / D[1], np.real(np.arccos(np.abs(V[0, D_sorted_indices[2]])))]
			
			JT['dur'].extend(jpk['end_time'] - jpk['start_time'])
			JT['dpth'].extend(Ps[np.clip(np.round(jpk['start_time'] * pfs).astype(int), 1, len(Ps) - 1)])
			JT['rmsJ'].extend(R[:, 0])
			JT['maxJ'].extend(R[:, 2])
			JT['maxA'].extend(R[:, 3])
			JT['maxAan'].extend(R[:, 4])
			JT['rmsMJ'].extend(R[:, 1])
			JT['maxMJ'].extend(R[:, 7])
			JT['cond'].extend(R[:, 5])
			JT['angle'].extend(R[:, 6])
		
		# JT['fstlst'] = pcas_in_dives(P, JT['time'])
		JT['sa'], JT['ttf'], JT['df'] = sunangle(depid, JT['time'])
		JT['THR'] = thr
		JT['BLNK'] = BLNK
		JT['MINDUR'] = MINDUR
		
		return JT
	
    def preprocess_time(self):
        """Preprocess time data and segment it if required."""
        if self.timeDays != 0:
            timeSinceStart = (self.timeRef - self.timeRef[0]).astype(float) / (24 * 60 * 60)
            refMax = np.floor(np.max(timeSinceStart) / self.timeDays)
            refBreak = np.arange(0, refMax + 1) * self.timeDays
            refBreak[-1] += self.timeDays

            timeBin = np.digitize(timeSinceStart, refBreak, right=True)
            return timeBin
        return None

    def filter_data(self):
        """Apply a high-pass Butterworth filter to the data."""
        bf_pca = signal.butter(3, self.fc / (0.5 * self.fs), btype='high', output='sos')
        self.filtered_x = signal.sosfiltfilt(bf_pca, self.x[:, 1:4], axis=0)

        # Handle NaN values
        nas = np.isnan(self.filtered_x)
        nas_vector = np.logical_or.reduce(nas, axis=1)
        if np.any(nas_vector):
            print("NAs found and replaced by 0. NA proportion:", np.mean(nas_vector))
            self.filtered_x[nas_vector, :] = 0
        
        # Smoothing using a moving average
        self.filtered_x = np.concatenate(
            (np.zeros((12, 3)), signal.sosfiltfilt(np.ones(25) / 25, self.filtered_x, axis=0), np.zeros((12, 3))),
            axis=0
        )

    def apply_kmeans(self):
        """Apply KMeans clustering to identify high states."""
        self.kmeans_model = KMeans(n_clusters=2).fit(self.filtered_x)
        high_state = np.argmax(self.kmeans_model.cluster_centers_, axis=0)
        logicalHigh = self.kmeans_model.labels_ == high_state
        self.filtered_x = logicalHigh.astype(float)

    def combine_data(self):
        """Combine the processed data."""
        x_combined = np.prod(self.filtered_x, axis=1)
        temp_x = np.column_stack((x_combined, self.timeRef))
        temp_x = np.max(temp_x, axis=0)
        return temp_x[:, np.newaxis]

    def process(self):
        """Run the full process and return the final result."""
        self.preprocess_time()
        self.filter_data()
        self.apply_kmeans()
        return self.combine_data()

import pandas as pd

class PreyCatchAttemptBehaviours:
    def __init__(self, x, x2, fs=16, fc=2.64, timeDays=0):
        """
        Initialize the object with necessary parameters and data.
        
        Args:
        - x: Pandas DataFrame containing columns 'time', 'ax', 'ay', 'az', 'segmentID'.
        - x2: segmentID identifying continuous time segments.
        - fs: Sampling frequency of signal (default is 16Hz).
        - fc: Frequency above which signals across x, y, and z axes are retained.
        - timeDays: Time period over which KMeans clustering should be performed.
                    Leave as zero to have no time grouping in KMeans generation.
        """
        if not isinstance(x, pd.DataFrame):
            raise ValueError("Input x must be a Pandas DataFrame")
        self.x = x
        self.x2 = x2
        self.fs = fs
        self.fc = fc
        self.timeDays = timeDays
        self.timeRef = self.x['time'].copy()

    def preprocess_time(self):
        """Preprocess the time data and create time bins if required."""
        if self.timeDays != 0:
            timeSinceStart = (self.timeRef - self.timeRef.iloc[0]).dt.total_seconds() / (24 * 60 * 60)
            refMax = np.floor(timeSinceStart.max() / self.timeDays)
            refBreak = np.arange(0, refMax + 1) * self.timeDays
            refBreak = np.append(refBreak, refBreak[-1] + self.timeDays)

            self.x['timeBin'] = np.digitize(timeSinceStart, refBreak, right=False)
            del refBreak, timeSinceStart

    def apply_high_pass_filter(self):
        """Apply a high-pass Butterworth filter to the data."""
        sos = signal.butter(3, self.fc / (0.5 * self.fs), btype='high', output='sos')
        
        def filter_func(col):
            return signal.sosfiltfilt(sos, col)
        
        self.x[['ax', 'ay', 'az']] = self.x.groupby(self.x2)[['ax', 'ay', 'az']].transform(filter_func)

        # Handle NaN values
        nas = self.x[['ax', 'ay', 'az']].isna()
        nas_vector = nas.any(axis=1)
        if nas_vector.any():
            print(f"NAs found and replaced by 0. NA proportion: {nas_vector.mean()}")
            self.x.loc[nas_vector, ['ax', 'ay', 'az']] = 0

    def smooth_data(self):
        """Apply a smoothing operation to the data using a rolling standard deviation."""
        def smooth_func(col):
            return np.concatenate((np.zeros(12), pd.Series(col).rolling(window=25, min_periods=1).std().fillna(0).values, np.zeros(12)))

        self.x[['ax', 'ay', 'az']] = self.x.groupby(self.x2)[['ax', 'ay', 'az']].transform(smooth_func)

    def apply_kmeans(self):
        """Apply KMeans clustering to classify data points."""
        def kmeans_func(group):
            km = KMeans(n_clusters=2)
            km.fit(group)
            high_state = np.argmax(km.cluster_centers_)
            return (km.labels_ == high_state).astype(float)
        
        if self.timeDays != 0:
            self.x[['ax', 'ay', 'az']] = self.x.groupby('timeBin')[['ax', 'ay', 'az']].transform(kmeans_func)
        else:
            self.x[['ax', 'ay', 'az']] = self.x[['ax', 'ay', 'az']].transform(kmeans_func)

    def combine_data(self):
        """Combine the processed data into a final result."""
        self.x['pca'] = self.x[['ax', 'ay', 'az']].prod(axis=1)
        result = self.x[['time', 'pca']].groupby('time').max().reset_index()
        return result

    def process(self):
        """Run the full process and return the final result."""
        self.preprocess_time()
        self.apply_high_pass_filter()
        self.smooth_data()
        self.apply_kmeans()
        return self.combine_data()

# Example usage
data = pd.DataFrame({
    'time': pd.date_range(start='2023-01-01', periods=100, freq='S'),
    'ax': np.random.randn(100),
    'ay': np.random.randn(100),
    'az': np.random.randn(100),
    'segmentID': np.repeat([1, 2, 3, 4], 25)
})

pcab = PreyCatchAttemptBehaviours(x=data, x2=data['segmentID'])
result = pcab.process()
print(result)

'''