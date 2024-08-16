import numpy as np
import scipy.signal as sg
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from scipy.fft import fft, fftfreq, fftshift
from SES_tag.utils.acoustic_utils import *
from SES_tag.wrapper import Wrapper


class Acoustic(Wrapper):
	"""
	Acoustic class for processing and analyzing acoustic data.

	This class provides methods to normalize acoustic data and compute the noise level
	using windowed Fourier transforms. It is initialized with a timestamp and parameters
	for data and spectrogram normalization.
	"""

	def __init__(self, timestamp, wav_path = None, *, data_normalization : str = 'instrument', spectro_normalization : str = 'density', **kwargs):
		"""
		Initializes the Acoustic class.

		Parameters
		----------
		timestamp: dataframe-like object
			A timestamp object, expected to contain file information for loading acoustic data. Stores beginning timestamp of each wavfile.
			Contains columns fn (filename)
		data_normalization : 'str', optional
			Type of data normalization to be applied. Default is 'instrument'.
		spectro_normalization : 'str', optional
			Type of spectrogram normalization to be applied. Default is 'density'.
		**kwargs: Additional parameters like 'window_size' (bins), 'nfft' (bins), and 'duration' (seconds).

		Attributes:
		self.timestamp: Stores wavefiles names and begin POSIX timestamps.
		self.samplerate: Sample rate extracted from the first file in the timestamp.
		self.params: Dictionary of parameters for window size, FFT size, offset and others.
		self.data_normalization: Method for data normalization.
		self.spectro_normalization: Method for spectrogram normalization.
		"""
		self.timestamp = timestamp
		self.wav_path = wav_path if wav_path else os.getcwd()
		self.samplerate = sf.info(self.timestamp.fn.iloc[0]).samplerate
		offset = self.dt * self.samplerate
		default = {window_size = 1024, nfft = 1024, offset = offset}
		self.params = {**default, **kwargs}
		if 'duration' in self.params.keys():
			self.params['window_size'] = self.params['duration'] * self.samplerate
		self.data_normalization = data_normalization
		self.spectro_normalization = spectro_normalization

	def __call__(self):
		return self.forward()

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
		final_freqs = None
		acoustic_data = []
		for i, row in self.timestamp.iterrows():
			data, fs = sf.read(os.path.join(self.wav_path, row.fn))
			data = self.normalize(data)
			data, freqs = self.compute_noise_level(data)

			if final_freqs is None:
				final_freqs = freqs
			else:
				if np.array_equal(final_freqs, freqs):
					acoustic_data.extend(data)
				else:
					print(f"Found different frequencies for file {row.fn}. Skipping.")

		if overwrite :
			for freq in freqs :
				if freqs in self.ds.variables:
					self.remove_variable(freq)

		for i, freq in enumerate(final_freqs) :
			if freq not in self.ds.variables:
				freq_dim = self.ds.createDimension(freq, len(acoustic_data))
				freq_var = self.ds.createVariable(freq, np.float32, (freq,))
				freq_var.units = 'dB re 1 uPa**2 / Hz' if self.spectro_normalization == 'density' else 'dB re 1 uPa**2'
				freq_var.db_type = 'absolute' if self.data_normalization == 'instrument' else 'relative'
				freq_var.long_name = f'Power spectral density at {freq} Hz' if self.spectro_normalization == 'density' else 'Power spectrum at {freq} Hz'
				freq_var[:] = acoustic_data[:, i]


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
		    data = (
		        (data * self.peak_voltage)
		        / (20 * np.log10(self.sensitivity / 1e6))
		        / 10 ** (self.gain_dB / 20))


	def compute_noise_level(self, data): 
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

		win = np.hamming(self.params['window_size'])
		if self.params['nfft'] < (self.params['window_size']):
		    if self.spectro_normalization == "density":
		        scale_psd = 1.0
		    if self.spectro_normalization == "spectrum":
		        scale_psd = 1.0
		else:
		    if self.spectro_normalization == "density":
		        scale_psd = 2.0 / (((win * win).sum()) * sample_rate)
		    if paself.spectro_normalization == "spectrum":
		        scale_psd = 2.0 / (win.sum() ** 2)

		Nbech = np.size(data)
		Nbwin = int((Nbech - self.params['window_size']) / self.params['offset'])
		freqs = np.fft.rfftfreq(self.params['nfft'], d=1 / sample_rate)

		Sxx = np.zeros([np.size(freqs), Nbwin])
		for idwin in range(Nbwin):
		    if self.nfft < (self.window_size):
		        x_win = data[idwin * self.params['offset'] : idwin * self.params['offset'] + self.window_size]
		        _, Sxx[:, idwin] = signal.welch(
		            x_win,
		            fs=sample_rate,
		            window="hamming",
		            nperseg=int(self.params['nfft']),
		            noverlap=int(self.params['nfft'] / 2),
		            scaling=self.params['spectro_normalization']
		        )
		    else:
		        x_win = data[idwin * self.params['offset'] : idwin * self.params['offset'] + self.window_size] * win
		        Sxx[:, idwin] = np.abs(np.fft.rfft(x_win, n=self.nfft)) ** 2
		    Sxx[:, idwin] *= scale_psd

		if self.data_normalization == "instrument":
		    log_spectro = 10 * np.log10((Sxx / (1e-12)) + (1e-20))

		if self.data_normalization == "zscore":
		    if self.spectro_normalization == "density":
		        Sxx *= sample_rate / 2  # value around 0dB
		        log_spectro = 10 * np.log10(Sxx + (1e-20))
		    if self.spectro_normalization == "spectrum":
		        Sxx *= self.params['window_size'] / 2  # value around 0dB
		        log_spectro = 10 * np.log10(Sxx + (1e-20))

		return log_spectro, freqs
		
		
