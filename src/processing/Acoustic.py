import numpy as np
import scipy.signal as sg
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from scipy.fft import fft, fftfreq, fftshift
from SES_tag.utils.acoustic_utils import *
from SES_tag.wrapper import Wrapper

class Acoustic(Wrapper):

	def __init__(self):
		
		default = {window_size = 1024, overlap = 20, nfft = 1024}
		self.params = {**default, **kwargs}
		self.data_normalization = data_normalization
		self.spectro_normalization = spectro_normalization
		pass

	def normalize(self):

	       if self.data_normalization == "instrument":
		    data = (
		        (data * self.peak_voltage)
		        / (20 * np.log10(self.sensitivity / 1e6))
		        / 10 ** (self.gain_dB / 20))


	def run_functions(size, step, timestamp, path, filename = 'spl_data.csv', iteration = 0):
		'''
		size : length of soundfile to analyze in seconds
		step : step between two successive sound files
		timestamp : contains beginning time of each audio file (epoch format)
		path : path to audio files eg '/run/..../ml20_293a/raw/'
		freqs : List of frequencies at which you want SPL to be computed
		'''
		timestamp = pd.read_csv(timestamp)
		spl_dict = {**{'time':np.arange(timestamp.begin.iloc[0], timestamp.end.iloc[-1], step)}}
		spl_dict['fns'], spl_dict['start_time'] = find_corresponding_file(timestamp, spl_dict['time'])
		data = []
		for i in tqdm(range(iteration*3600, len(spl_dict['fns']) - iteration*3600), position = 0, leave = True):
			fs = sf.info(path+spl_dict['fns'][i]).samplerate
				sig, fs = sf.read(path+spl_dict['fns'][i], start = int((spl_dict['time'][i]-spl_dict['start_time'][i])*fs), stop = int((spl_dict['time'][i]-spl_dict['start_time'][i]+size)*fs))
			f, Pxx = sg.welch(sig, fs=fs, nperseg=2048)
			_row = np.append(Pxx, spl_dict['time'][i])
			data.append(_row)
			if (i+1) % 3600 == 0:
				df = pd.DataFrame(data)
				df.columns = np.append(f, ['time'])
				df.to_csv(f'{i//3600}_{filename}')
				data = []

	def osmose(self): 

		Noverlap = int(self.window_size * self.overlap / 100)

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
		Noffset = self.params['window_size'] - Noverlap
		Nbwin = int((Nbech - self.params['window_size']) / Noffset)
		Freq = np.fft.rfftfreq(self.params['nfft'], d=1 / sample_rate)

		Sxx = np.zeros([np.size(Freq), Nbwin])
		Time = np.linspace(0, Nbech / sample_rate, Nbwin)
		for idwin in range(Nbwin):
		    if self.nfft < (self.window_size):
		        x_win = data[idwin * Noffset : idwin * Noffset + self.window_size]
		        _, Sxx[:, idwin] = signal.welch(
		            x_win,
		            fs=sample_rate,
		            window="hamming",
		            nperseg=int(self.params['nfft']),
		            noverlap=int(self.params['nfft'] / 2),
		            scaling=self.params['spectro_normalization']
		        )
		    else:
		        x_win = data[idwin * Noffset : idwin * Noffset + self.window_size] * win
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

			
		
		
