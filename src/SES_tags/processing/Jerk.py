import numpy as np
from glob import glob
import os
from SES_tags.wrapper import Wrapper
import netCDF4 as nc
from datetime import datetime
from SES_tags.utils.inertial_utils import * 
from SES_tags.utils.format_utils import *
import soundfile as sf
from scipy.interpolate import interp1d

class Jerk(Wrapper):
	
	threshold = 200
	blanking = 5
	duration = None
	
	def __init__(self, 
			  depid, 
			  *,
			  path,
			  sens_path,
			  data = {'time': None, 'jerk' : None, 'P' : None}
			  ) :
		
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
			self.sens_time = datetime.strptime(data.dephist_deploy_datetime_start, '%Y/%m/%d %H:%M:%S').timestamp() + np.arange(0, length/self.samplerate, np.round(1/self.samplerate,2))
			self.A_cal_poly = data['A'].cal_poly[:].reshape(2, 3)
			self.A_cal_map = data['A'].cal_map[:].reshape(3, 3)
			self.M_cal_poly = data['M'].cal_poly[:].reshape(-1, 3)
			self.M_cal_cross = data['M'].cal_cross[:].reshape(-1, 3)
			self.M_cal_tseg = data['M'].cal_tseg[:]
			self.M_cal_map = data['M'].cal_map[:].reshape(3, 3)
			
		elif data['A'] is not None and data['M'] is not None and data['time'] is not None :
			self.sens_time, self.jerk, self.P = data['time'], data['jerk'], data['P']
			self.samplerate = np.round(1 / (self.sens_time[1]-self.sens_time[0]), 2)
		
		#Make P the same length as jerk
		self.P = np.pad(self.P[:len(self.jerk)], (0, max(0, len(self.jerk) - len(self.P))), constant_values=np.nan)
		
		#Remove surface data
		self.jerk[self.P <= 20] = np.nan


	def __call__(self) :
		self.get_peaks()
		
	def check_fullbw(self, raw_path, hr_threshold=400) :
		swv_fns = np.array(glob(os.path.join(raw_path, '*swv')))
		xml_fns = np.array(glob(os.path.join(raw_path, '*xml')))
		xml_fns = xml_fns[xml_fns != glob(os.path.join(raw_path, '*dat.xml'))]
		xml_start_time = get_start_time_xml(xml_fns)
		for i, peak_time in enumerate(self.peaks['timestamp']) :
			fn_idx = np.argmax(xml_start_time[xml_start_time - peak_time < 0])
			fs = sf.info(swv_fns[fn_idx]).samplerate
			sig, fs = sf.read(swv_fns[fn_idx], 
					 start = int((self.sens_time[0] + self.peaks['start_time'][i] - xml_start_time[fn_idx] - 1)*fs),
					 stop = int((self.sens_time[0] + self.peaks['end_time'][i] - xml_start_time[fn_idx] + 1)*fs))
			A_peak = sig[:, :3]
			A_peak = (A_peak * self.A_cal_poly[0] + self.A_cal_poly[1]) @ self.A_cal_map
			jerk_peak = self.njerk(A_peak, fs)
			depth_interp = interp1d(self.sens_time, self.P, bounds_error=False, fill_value=np.nan)
			depth_peak = depth_interp(np.linspace(self.sens_time[0] + self.peaks['start_time'][i]-1, 
										self.sens_time[0] + self.peaks['end_time'][i]+1, 
										len(sig)))
			jerk_peak[depth_peak <= 20] = np.nan
			jerk_validation = self.get_peaks(jerk = jerk_peak,
									samplerate = fs,
									threshold = hr_threshold,
									blanking = 0.25,
									duration = 0.02)			

	def get_peaks(self, **kwargs):
		"""
		Determine the start sample of peaks that are above the threshold.

		Parameters :
		----------
		x : numpy array
			Input signal.
		fs : float
			Sampling frequency.
		thresh : float
			Threshold value to detect peaks.
		blanking : int or float
			Blanking criterion in seconds. 
		duration : int or float, optional
			Minimum duration of the peak in seconds. Default value is None.
		"""
		
		self.peaks = {}
		default = {'jerk':self.jerk, 'blanking':self.blanking, 'samplerate':self.samplerate, 'threshold':self.threshold, 'duration':self.duration}
		params = {**default, **kwargs}
		self.peaks['blanking'] = params['blanking']
		#Go from seconds to number of bins
		params['blanking'] *= params['samplerate']
		
		#Find jerk peaks above threshold
		dxx = np.diff((params['jerk'] >= params['threshold']).astype(int))
		cc = np.where(dxx > 0)[0] + 1
		if len(cc) == 0:
			return None
		
		# Find ending sample of each peak
		coff = np.where(dxx < 0)[0] + 1
		cend = np.full(len(cc), len(params['jerk']))
		for k in range(len(cc)):
			kends = np.where(coff > cc[k])[0]
			if len(kends) > 0:
				cend[k] = coff[kends[0]]
		
		# Eliminate detections which do not meet blanking criterion & merge pulses that are within blanking distance
		done = False
		while not done:
			kg = np.where(cc[1:] - cend[:-1] > params['blanking'])[0]
			done = len(kg) == (len(cc) - 1)
			cc = cc[np.concatenate(([0], kg + 1))]
			cend = cend[np.concatenate((kg, [len(cend) - 1]))]
		if cend[-1] == len(params['jerk']):
			cc = cc[:-1]
			cend = cend[:-1]
		
		# Remove peaks shorter than duration attribute
		if params['duration'] :
			params['duration'] *= params['samplerate']
			k = np.where(cend - cc >= params['duration'])[0]
			cc = cc[k]
			cend = cend[k]
			minlen = params['duration'] / params['samplerate']
		else:
			minlen = 1 / params['samplerate']
		
		# Determine the time and maximum of each peak
		peak_time = np.zeros(len(cc))
		peak_max = np.zeros(len(cc))
		for a in range(len(cc)):
			segment = params['jerk'][cc[a]:cend[a]]
			index = np.argmax(segment)
			peak_time[a] = index + cc[a]
			peak_max[a] = np.max(segment)
		
		self.peaks['start_time'] = cc / params['samplerate']  #in seconds
		self.peaks['end_time'] = cend / params['samplerate']  #in seconds
		self.peaks['maxtime'] = peak_time / params['samplerate'] #in seconds
		self.peaks['max'] = peak_max
		self.peaks['threshold'] = params['threshold']
		self.peaks['minlen'] = minlen
		self.peaks['duration'] = self.peaks['end_time'] - self.peaks['start_time']
		self.peaks['depth'] = self.P[cc]
		peak_times = self.sens_time[0] + self.peaks['start_time']
		self.peaks['datetime'] = np.array(list(map(lambda x : datetime.fromtimestamp(x), peak_times)))
		self.peaks['timestamp'] =  np.array(list(map(lambda x : x.timestamp(), self.peaks['datetime'])))

	# Function taken from animaltags Python package
	@staticmethod
	def njerk(A, sampling_rate):
		if isinstance(A, dict):
			sampling_rate = A["sampling_rate"]
			a = A["data"]
			j = A.copy()
			j["data"] = np.concatenate((sampling_rate * np.sqrt(np.sum(np.diff(a, axis=0)**2, axis=1)), [0]))
			j["creation_date"] = datetime.now().isoformat()
			j["type"] = "njerk"
			j["full_name"] = "norm jerk"
			j["description"] = j["full_name"]
			j["unit"] = "m/s3"
			j["unit_name"] = "meters per seconds cubed"
			j["unit_label"] = "m/s^3"
			j["column_name"] = "jerk"
		else:
			a = A
			j = sampling_rate * np.concatenate((np.sqrt(np.sum(np.diff(a, axis=0)**2, axis=1)), [0]))
		
		return j
	
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