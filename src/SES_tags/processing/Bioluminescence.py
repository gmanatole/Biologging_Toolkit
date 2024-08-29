import numpy as np
import scipy.io
import netCDF4 as nc
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from SES_tags.wrapper import Wrapper
from SES_tags.utils.format_utils import *
from glob import glob
import os

class Bioluminescence(Wrapper):

	def __init__(self):
		self.samplerate = 5
		pass

	def forward(self) :

		fsin = 50   #Origin sampling rate
		
		if fsin % self.samplerate != 0:
			print(f"Output fs must be an integer divisor of raw sampling rate ({fsin} Hz)")
			return []
		
		bl = round(fsin / self.samplerate)  # Calculate sampling rate ratio
		nsamps = bl * round(LEN * fsin / bl)  # Calculate number of samples in a block
		len_ = nsamps / fsin  # Length of block in seconds
		lenreq = len_ + bl / fsin
		cue = 0
		
		L = []
		
		swv_fns = np.array(glob(os.path.join(raw_path, '*swv')))
		for fn in swv_fns :
			sig, fs = sf.read(fn)
			ll = sig[:, get_xml_columns(fn[:-3] + 'xml')]
		
			# Clean the data
			ll = self.fix_light_data(ll)
		
			# Buffer the data into blocks
			n_blocks = len(ll) // bl
			Y = np.array([ll[i:i+bl] for i in range(0, n_blocks * bl, bl)])
		
			# Calculate the max of each block and append to L
			L.extend(np.max(Y, axis=1))
		
			# Ensure L is a column vector
			L = np.array(L)

			if len(L) > 0:
				L = np.append(L, L[-1])  # Add one measurement to equalize length of other sensors
		
		return L
	
	
	def clean_data(self, ll):
		L = 0
		ll = self.fix_light_sens(ll)
		nbl = int(np.ceil(n / INTVL))

		# Ensure ll is not longer than nsamps
		if len(ll) > nbl:
			ll = ll[:nbl]
		        
		# Buffer the data (equivalent of MATLAB's buffer function)
		Y = np.array([ll[i:i + bl] for i in range(0, len(ll) - bl + 1, bl)]).T
		        
		 # Calculate the max of each column and append to L
		L.extend(np.max(Y, axis=0))
		        

    
	def fix_light_data(self, L) :
		
		INTVL = 101  # light interference cycle length in samples at 50 Hz
		
		# Find indices where L > 0.99
		kc = np.where(L > 0.99)[0]
		L[kc] = np.nan
		
		# Buffer the data into columns of length INTVL to compute median
		n = len(L)
		nbl = int(np.ceil(n / INTVL))
		pad_length = (INTVL - (n % INTVL)) % INTVL
		L_padded = np.pad(L, (0, pad_length), constant_values=np.nan)  #To create 2D matrix of size (-1, INTVL)
		Lb = L_padded.reshape(-1, INTVL).T
		Lp = np.nanmedian(Lb, axis=1)
		
		# Subtract the interference pattern from the original signal
		L_corrected = L - np.tile(Lp, nbl)[:n]
		L_corrected = np.append(L_corrected, Lp[:n - len(L_corrected)])
		
		# Apply a median filter with a kernel size of 3
		L_corrected = medfilt(np.abs(L_corrected), kernel_size=3)
		
		# Restore the original high values
		L_corrected[kc] = 1 - np.nanmedian(Lp)



'''    def get_gain_times(self):
        """Get gain times and save them."""
        self.T, self.G = self.get_ext_gain(self.recdir, self.depid)
        scipy.io.savemat(f'{self.depid}_gain_times_TG.mat', {'T': self.T, 'G': self.G})

    def downsample_light_data(self, downsample_factor=5):
        """Downsample light data to the specified frequency."""
        self.LL = self.d3maxlight(self.recdir, self.depid, downsample_factor)

    def load_pressure_data(self):
        """Load pressure data from the .nc file."""
        filepath = f'{self.drive}nc_files/{self.depid}sens5.nc'
        self.P = self.load_nc(filepath, ['P'])
        self.LL['data'][self.P['data'] < 20] = np.nan  # Remove surface data
        self.P['sampling_rate'] = self.P['samplingrate']

    def crop_and_plot(self, time_range):
        """Crop the data to the specified time range and plot."""
        P2 = self.crop_to(self.P, time_range)
        L2 = self.crop_to(self.LL, time_range)
        self.plott(P2, L2)
        del P2, L2

    def keep_only_gain_3(self):
        """Filter data to keep only gain 3."""
        mat_data = scipy.io.loadmat(f'{self.depid}_gain_times_TG.mat')
        self.T, self.G = mat_data['T'], mat_data['G']
        t = self.gps2tag_time(self.T, self.get_info())

        k = np.where(self.G == 3)[0]
        if self.G[-1] == 3:
            k = k[:-1]

        t_start = t[k]
        t_end = t[k + 1]

        ton = t_start + 60  # Allow 60 seconds for the sensor to settle
        dur = t_end - ton
        k, nk = self.eventon(np.vstack((ton, dur)).T, np.arange(len(self.LL['data'])) / self.LL['sampling_rate'])
        self.LL['data'][nk] = np.nan

        self.crop_and_plot([15, 20] * 3600 * 24)

    def high_pass_filter_light_data(self):
        """High pass filter the light data to remove low-frequency noise."""
        cutoff = 0.2 / (self.LL['sampling_rate'] / 2)
        self.LL['data_no_filter'] = self.LL['data']
        self.LL['data'] = np.abs(self.fir_nodelay(self.LL['data'], self.LL['sampling_rate'], cutoff, 'high'))

        self.crop_and_plot([15, 20] * 3600 * 24)

    def save_light_data(self):
        """Save the filtered light data to a NetCDF file."""
        self.add_nc(f'{self.drive}nc_files/{self.depid}sens5.nc', self.LL)

    @staticmethod
    def get_ext_gain(recdir, depid):
        # Implement this function based on your MATLAB code
        pass

    @staticmethod
    def d3maxlight(recdir, depid, downsample_factor):
        # Implement this function based on your MATLAB code
        pass

    @staticmethod
    def load_nc(filepath, variables):
        ds = nc.Dataset(filepath)
        data = {var: ds.variables[var][:] for var in variables}
        ds.close()
        return data

    @staticmethod
    def crop_to(data, time_range):
        # Implement this function based on your MATLAB code
        pass

    @staticmethod
    def plott(P, L):
        # Implement this function based on your MATLAB code
        pass

    @staticmethod
    def gps2tag_time(T, info):
        # Implement this function based on your MATLAB code
        pass

    @staticmethod
    def eventon(events, time_series):
        # Implement this function based on your MATLAB code
        pass

    @staticmethod
    def fir_nodelay(data, sampling_rate, cutoff, filter_type):
        numtaps = 10 * int(sampling_rate)
        taps = firwin(numtaps, cutoff, pass_zero=filter_type)
        return lfilter(taps, 1.0, data)

    @staticmethod
    def add_nc(filepath, data):
        # Implement this function to save data to the NetCDF file
        pass

    @staticmethod
    def get_info():
        # Retrieve the 'info' data structure as used in the original MATLAB code
        pass'''
