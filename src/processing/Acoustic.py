import numpy as np
import scipy.signal as sg
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from scipy.fft import fft, fftfreq, fftshift
from SES_tag.utils import *

class Acoustic():

	def __init__(self):
		pass

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

			
		
		
