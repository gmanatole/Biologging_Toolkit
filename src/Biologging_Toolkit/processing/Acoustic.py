import numpy as np
import os
import scipy.signal as sg
import soundfile as sf
from tqdm import tqdm
from SES_tags.utils.acoustic_utils import *
from SES_tags.utils.format_utils import *
from SES_tags.wrapper import Wrapper
from glob import glob
import pdb
import gc
from joblib import Parallel, delayed

class Acoustic(Wrapper):
	"""
	Acoustic class for processing and analyzing acoustic data.

	This class provides methods to normalize acoustic data and compute the noise level
	using windowed Fourier transforms. It is initialized with a timestamp and parameters
	for data and spectrogram normalization.
	"""

	def __init__(self,
			  depid,
			  *,
			  path,
			  wav_path = None,
			  data_normalization : str = 'instrument',
			  spectro_normalization : str = 'density',
			  **kwargs
			  ):
		"""
		Initializes the Acoustic class.
		For instrument normalization please input an instrument dictionary with peak_voltage, gain_dB and sensitivity keys. Values should be positive.

		Parameters
		----------
		timestamp: dataframe-like object
			A timestamp object, expected to contain file information for loading acoustic data. Stores beginning and ending POSIX timestamp of each wavfile.
			Contains columns fn (filename)
		data_normalization : 'str', optional
			Type of data normalization to be applied. Either 'instrument' or 'zscore'. Default is 'instrument'.
		spectro_normalization : 'str', optional
			Type of spectrogram normalization to be applied. Either 'density' or 'spectrum'. Default is 'density'.
		**kwargs: Additional parameters like 'window_size' (bins), 'nfft' (bins), and 'duration' (seconds).

		Attributes:
		self.timestamp: Stores wavefiles names and begin POSIX timestamps.
		self.samplerate: Sample rate extracted from the first file in the timestamp.
		self.params: Dictionary of parameters for window size, FFT size, offset and others.
		self.data_normalization: Method for data normalization.
		self.spectro_normalization: Method for spectrogram normalization.
		"""
		

		super().__init__(
			depid,
			path
        )
		
		self.wav_path = wav_path
		self.get_timestamps()
		
		self.samplerate = sf.info(os.path.join(self.wav_fns[0])).samplerate
		default = {'window_size' : 1024, 'nfft' : 1024, 'overlap' : 0, 'duration' : 3}
		self.params = {**default, **kwargs}

		self.data_normalization = data_normalization
		if self.data_normalization == 'instrument':
			# Check that 'instrument' exists and is a dictionary
			assert 'instrument' in self.params, "Please make sure to provide the hydrophone's sensitivity, peak_voltage and gain_dB value in a dictionary named instrument.\nElse choose zscore"
			assert isinstance(self.params['instrument'], dict), "'instrument' is not a dictionary."
			
			# Check that required keys exist in the 'instrument' dictionary
			required_keys = ['gain_dB', 'sensitivity', 'peak_voltage']
			for key in required_keys:
				assert key in self.params['instrument'], f"'{key}' key is missing in 'instrument' dictionary."
				assert self.params['instrument'][key] is not None, f"'{key}' key has no value in 'instrument' dictionary."

		self.spectro_normalization = spectro_normalization


	def __call__(self, overwrite = False):
		return self.forward(overwrite = overwrite)


	def forward(self, overwrite = False) :
		"""
		Processes acoustic data files, normalizes them, computes noise levels, and updates a NetCDF dataset with the results.
		Parameters
		----------
		overwrite : bool
		A flag indicating whether to overwrite existing variables in the NetCDF dataset.

		Raises:
		------
		ValueError:
		If frequency components (`freqs`) differ across files.
		"""
		
		log_spectro, freqs = self.compute_noise_level()
		
		# Create or retrieve the time dimension (assuming it's already in the dataset)
		if 'time' not in self.ds.dimensions:
			raise ValueError("Time dimension is missing from the dataset. Please call create_time method first.")
			
		# Remove existing spectrogram variable if overwriting
		if overwrite :
			if 'spectrogram' in self.ds.variables:
				self.remove_variable('spectrogram')
			if 'frequency_spectrogram' in self.ds.variables:
				self.remove_variable('frequency_spectrogram')
		
		# Create or retrieve the frequency dimension and variable
		freq_dim = self.ds.createDimension('frequency_spectrogram', log_spectro.shape[0])
		freq_var = self.ds.createVariable('frequency_spectrogram', np.float32, ('frequency_spectrogram',))
		freq_var.units = 'Hz'
		freq_var.long_name = 'Frequency values'
		freq_var[:] = freqs

		# Create or update the spectrogram variable
		spectrogram_var = self.ds.createVariable('spectrogram', np.float32, ('time', 'frequency_spectrogram'))
		spectrogram_var.units = 'dB re 1 uPa**2 / Hz' if self.spectro_normalization == 'density' else 'dB re 1 uPa**2'
		spectrogram_var.db_type = 'absolute' if self.data_normalization == 'instrument' else 'relative'
		spectrogram_var.long_name = 'Spectrogram data'

		spectrogram_var[:] = log_spectro.T


	def normalize(self, data) :
		"""
		Normalizes the input acoustic data based on the specified data_normalization method.
		Parameters
		----------
		    data: array-like
			The raw acoustic data to be normalized.
		Returns
		-------
		    Normalized data.
		"""
		if self.data_normalization == "instrument":
			assert self.params['instrument']['sensitivity'] > 0, 'Please make sure sensitivity value is positive'
			data = (
		        (data * self.params['instrument']['peak_voltage'])
		        / (20 * np.log10(self.params['instrument']['sensitivity'] / 1e6))
		        / 10 ** (self.params['instrument']['gain_dB'] / 20))
		return data
		
	def compute_noise_level(self): 
		"""
		Computes the noise level of the provided acoustic data using windowed Fourier transforms.
		Parameters
		----------
		data: array-like
			The raw acoustic data for which the noise level is to be computed.
		Returns
		-------
		log_spectro: The computed and normalized spectrogram, representing the noise level in the data.
		"""

		# Create scales wrt to user parameters
		win = np.hamming(self.params['window_size'])
		if self.params['nfft'] < (self.params['window_size']):
		    if self.spectro_normalization == "density":
		        scale_psd = 1.0
		    if self.spectro_normalization == "spectrum":
		        scale_psd = 1.0
		else:
		    if self.spectro_normalization == "density":
		        scale_psd = 2.0 / (((win * win).sum()) * self.samplerate)
		    if self.spectro_normalization == "spectrum":
		        scale_psd = 2.0 / (win.sum() ** 2)

		# Get FFT parameters
		#Nbech = np.size(data)
		#Nbwin = int((Nbech - self.params['offset']) / self.params['offset'])
		freqs = np.fft.rfftfreq(self.params['nfft'], d=1 / self.samplerate)
		Noverlap = int(self.params['window_size'] * self.params['overlap'] / 100)
		Nbech = self.params['duration'] * self.samplerate
		Noffset = self.params['window_size'] - Noverlap
		Nbwin = int((Nbech - self.params['window_size']) / Noffset)
		Nfreqs = np.size(freqs)
		
		# Fetch wav starting times wrt to netCDF structure
		matches = (self.ds['time'][:].data[:, None] >= self.wav_start_time) & (self.ds['time'][:].data[:, None] <= self.wav_end_time)
		indices = np.where(matches.any(axis=1), matches.argmax(axis=1), -1)
		time_diffs = np.where(indices != -1, self.ds['time'][:].data - self.wav_start_time[indices], np.nan)
		spectro = np.full([np.size(freqs), len(indices)], np.nan)
		
		pbar = tqdm(total = len(self.wav_start_time), leave = True, position = 0)
	
		for idx, wav_file in enumerate(self.wav_fns[:10]) :

			pbar.update(1)
			pbar.set_description(f'Computing spectral power for file : {wav_file}')
			
			# Fetch data corresponding to one wav file
			data, _ = sf.read(wav_file, dtype = 'float32')
			data = self.normalize(data)
			_time_diffs = time_diffs[indices == idx]
			
			# Read signal at correct timestamp
			for j in range(len(_time_diffs)):
				'''sig, _ = sf.read(wav_file,
					 start = int(_time_diffs[j] * self.samplerate),
					 stop =  int((_time_diffs[j] + self.params['duration']) * self.samplerate),
					 dtype = 'float32'
					 )'''
				sig = data[int(_time_diffs[j] * self.samplerate) : int((_time_diffs[j] + self.params['duration']) * self.samplerate)]
				sig = self.normalize(sig)
				Sxx = np.zeros([Nfreqs, Nbwin])
				#Compute the spectrogram for desired duration and with chosen window parameters
				for idwin in range(Nbwin):
					if self.params['nfft'] < (self.params['window_size']):
						x_win = sig[idwin * Noffset : idwin * Noffset + self.params['window_size']]
						_, Sxx[:, idwin] = sg.welch(
							x_win,
							fs=self.samplerate,
							window="hamming",
							nperseg=int(self.params['nfft']),
							noverlap=int(self.params['nfft'] / 2),
							scaling=self.spectro_normalization,
							)
					else:
						x_win = sig[idwin * Noffset : idwin * Noffset + self.params['window_size']]
						if len(x_win) < self.params['window_size'] :
							x_win = np.pad(x_win, (0, self.params['window_size']-len(x_win)), 'constant', constant_values=0)
						x_win = x_win * win
						Sxx[:, idwin] = np.abs(np.fft.rfft(x_win, n=self.params['nfft'])) ** 2
					Sxx[:, idwin] *= scale_psd
				spectro[:, np.argmax(indices == idx) + j] = np.mean(Sxx, axis = 1)			
				del sig, Sxx
				gc.collect()
				
			del data
			gc.collect()
			
			pbar.set_description('Normalizing and saving data')
			
		# Normalize spectrogram
		if self.data_normalization == "instrument":
		    log_spectro = 10 * np.log10((spectro / (1e-12)) + (1e-20))

		if self.data_normalization == "zscore":
		    if self.spectro_normalization == "density":
		        spectro *= self.samplerate / 2  # value around 0dB
		        log_spectro = 10 * np.log10(spectro + (1e-20))
		    if self.spectro_normalization == "spectrum":
		        spectro *= self.params['window_size'] / 2  # value around 0dB
		        log_spectro = 10 * np.log10(spectro + (1e-20))
				
		return log_spectro, freqs
		
	def get_timestamps(self) :
		self.wav_fns = np.array(glob(os.path.join(self.wav_path, '*wav')))
		xml_fns = np.array(glob(os.path.join(self.wav_path, '*xml')))
		xml_fns = xml_fns[xml_fns != glob(os.path.join(self.wav_path, '*dat.xml'))]
		self.wav_start_time = get_start_date_xml(xml_fns)
		wav_end_time = []
		for file in self.wav_fns :
			wav_end_time.append(sf.info(file).duration)
		wav_end_time = np.array(wav_end_time) + self.wav_start_time
		self.wav_end_time = wav_end_time
	