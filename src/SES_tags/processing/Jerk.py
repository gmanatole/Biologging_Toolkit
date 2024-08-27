import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
from SES_tags.wrapper import Wrapper
import netCDF4 as nc
from datetime import datetime

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
			self.jerk = data['J'][:].data
			self.P = data['P'][:].data
			self.sens_time = datetime.strptime(data.dephist_deploy_datetime_start, '%Y/%m/%d %H:%M:%S').timestamp() + np.arange(0, len(self.jerk)/5, 0.2)
		elif data['A'] is not None and data['M'] is not None and data['time'] is not None :
			self.sens_time, self.jerk, self.P = data['time'], data['jerk'], data['P']
		self.samplerate = 1 / (self.sens_time[1]-self.sens_time[0])
		
		#Make P the same length as jerk
		self.P = np.pad(self.P[:len(self.jerk)], (0, max(0, len(self.jerk) - len(self.P))), constant_values=np.nan)
		
		#Remove surface data
		self.jerk[self.P >= 20] = np.nan
	
	
	def forward(self):
		self.get_peaks()
		peak_times = self.sens_time[0] + self.peaks['start_time']
		self.peaks['datetime'] = list(map(peak_times, lambda x : datetime.fromtimestamp(x)))


	# Function taken from animaltags Python package
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

	def get_peaks(self):
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
		self.peaks['bktime'] = self.blanking

		#Go from seconds to number of bins
		self.blanking *= self.samplerate
		
		#Find jerk peaks above threshold
		dxx = np.diff((self.jerk >= self.threshold).astype(int))
		cc = np.where(dxx > 0)[0] + 1
		
		# Find ending sample of each peak
		coff = np.where(dxx < 0)[0] + 1
		cend = np.full(len(cc), len(self.jerk))
		for k in range(len(cc)):
			kends = np.where(coff > cc[k])[0]
			if len(kends) > 0:
				cend[k] = coff[kends[0]]
		
		# Eliminate detections which do not meet blanking criterion & merge pulses that are within blanking distance
		done = False
		while not done:
			kg = np.where(cc[1:] - cend[:-1] > self.blanking)[0]
			done = len(kg) == (len(cc) - 1)
			cc = cc[np.concatenate(([0], kg + 1))]
			cend = cend[np.concatenate((kg, [len(cend) - 1]))]
		
		if cend[-1] == len(self.jerk):
			cc = cc[:-1]
			cend = cend[:-1]
		
		# Remove peaks shorter than duration attribute
		if self.duration :
			self.duration *= self.samplerate
			k = np.where(cend - cc >= self.duration)[0]
			cc = cc[k]
			cend = cend[k]
			minlen = self.duration / self.samplerate
		else:
			minlen = 1 / self.samplerate
		
		# Determine the time and maximum of each peak
		peak_time = np.zeros(len(cc))
		peak_max = np.zeros(len(cc))
		for a in range(len(cc)):
			segment = self.jerk[cc[a]:cend[a]]
			index = np.argmax(segment)
			peak_time[a] = index + cc[a]
			peak_max[a] = np.max(segment)
		
		self.peaks['start_time'] = cc / self.samplerate  #in seconds
		self.peaks['end_time'] = cend / self.samplerate  #in seconds
		self.peaks['maxtime'] = peak_time / self.samplerate  #in seconds
		self.peaks['max'] = peak_max
		self.peaks['thresh'] = self.threshold
		self.peaks['minlen'] = minlen
		self.peaks['duration'] = self.peaks['end_time'] - self.peaks['start_time']
		self.peaks['depth'] = self.P[cc]

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