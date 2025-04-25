import numpy as np
import netCDF4 as nc
from scipy.signal import medfilt, lfilter, firwin
from scipy.interpolate import interp1d
from Biologging_Toolkit.wrapper import Wrapper
from Biologging_Toolkit.applications.Jerk import Jerk
from Biologging_Toolkit.utils.format_utils import *
from Biologging_Toolkit.auxiliary.Sun import *
from glob import glob
import os
import soundfile as sf

class Bioluminescence(Wrapper):

	INTVL = 101  # light interference cycle length in samples at 50 Hz
	
	def __init__(self,
			  depid,
			  *,
			  path,
			  sens_path : str = None,
			  trk_path : str = None,
			  raw_path : str = None,
			  samplerate = None
			  ):

		super().__init__(depid,path)

		if (sens_path is not None) & (trk_path is not None) :
			sens = nc.Dataset(sens_path)
			trk = nc.Dataset(trk_path)
			self.samplerate, start = sens['LL'].sampling_rate, get_start_time_sens(sens.dephist_device_datetime_start)
			self.LL = sens['LL'][:].data
			self.biolum_time = np.linspace(0, len(self.LL), len(self.LL))/self.samplerate+start
			lat, lon, trk_time = trk['POS'][:].data[1], trk['POS'][:].data[2], get_time_trk(trk)
			self.lat, self.lon = interp1d(trk_time, lat, bounds_error=False)(self.biolum_time), interp1d(trk_time, lon, bounds_error=False)(self.biolum_time)
		self.samplerate = self.dt if not samplerate else samplerate
		self.raw_path = raw_path

	def __call__(self, overwrite = False):
		self.forward(overwrite)

	def forward(self, overwrite = False):
		ds_flash = []
		for t in self.ds['time'] :
			start, end = t, t + self.dt
			mask = (self.biolum_time >= start) & (self.biolum_time < end)
			_intensity = self.flash_intensity[mask]
			_duration = self.flash_duration[mask]
			if len(_intensity) > 0:
				ds_flash.append([np.nanmax(_intensity), np.nanmax(_duration)])
			else:
				ds_flash.append([np.nan, np.nan])

		start_times = self.ds['time'][:].data[:-1]
		end_times = self.ds['time'][:].data[1:]
		bin_indices = np.searchsorted(end_times, self.biolum_time, side='right') - 1
		valid = (bin_indices >= 0) & (self.biolum_time >= start_times[bin_indices])
		max_intensity = np.full(len(self.ds['time']), np.nan)
		max_duration = np.full(len(self.ds['time']), np.nan)
		for i in np.unique(bin_indices[valid]):
			mask = bin_indices == i
			max_intensity[i] = np.nanmax(self.flash_intensity[mask])
			max_duration[i] = np.nanmax(self.flash_duration[mask])
		ds_flash = np.stack([max_intensity, max_duration], axis=1)

		if overwrite:
			if 'flash' in self.ds.variables:
				self.remove_variable('flash')

		if 'flash' not in self.ds.variables:
			flash = self.ds.createVariable('flash', np.float64, ('time', 'feature'))
			flash.long_name = f'Maximum flash intensity and duration'
			flash.feature_description = '0: intensity, 1: duration'
			flash.raw_data_samplerate = self.raw_samplerate
			flash.threshold = self.threshold
			flash.intensity_comment = 'Maximum intensity of flashes'
			flash.duration_comment = 'Maximum duration of peaks'
			flash.duration_units = 's'
			flash[:,:] = ds_flash

	def get_flash(self, threshold = 0.98, sun_threshold = -12):
		self.threshold = threshold
		_sun = np.pi/2 - get_sun_pos(self.biolum_time[::40], self.lat[::40], self.lon[::40])[1]
		self.sun = interp1d(self.biolum_time[::40], _sun, fill_value='extrapolate')(self.biolum_time)
		groups = np.zeros(len(self.sun))
		groups[self.sun < sun_threshold * np.pi / 180] = -1
		groups[self.sun > 0] = 1
		self.flash_duration, self.flash_intensity = np.zeros(len(self.LL)), np.zeros(len(self.LL))
		for lum in [-1,0,1] :
			_LL = self.LL.copy()
			_LL[groups != lum] = 0
			_thresh = np.nanquantile(self.LL[groups == lum], threshold)
			pos_start, pos_end, flash_time, flash_max, min_len = Jerk.get_peaks(_LL, samplerate = 5, blanking = 1, threshold = _thresh)
			self.flash_duration[pos_start] = (pos_end - pos_start) / 5
			self.flash_intensity[pos_start] = flash_max

	def process_raw(self, fsin = 50, length = 3600) :
		'''
		Parameters
		----------
		fsin : sampling rate
		length : length of block to analyze
		Returns
		-------
		Light level for all gains from swv files
		'''
		#ADD ARGUMENT GAIN FLAG TO ONLY DO IT ON GAIN 3 OR ON ALL THE GAIN
		if fsin % self.samplerate != 0:
			print(f"Output fs must be an integer divisor of raw sampling rate ({fsin} Hz)")
			return []
		
		bl = round(fsin / self.samplerate)  # Calculate sampling rate ratio
		nsamps = bl * round(length * fsin / bl)  # Calculate number of samples in a block
		len_ = nsamps / fsin  # Length of block in seconds
		lenreq = len_ + bl / fsin
		cue = 0
		swv_fns = np.array(glob(os.path.join(self.raw_path, '*swv')))
		L = []
		L_col = get_xml_columns(swv_fns[0][:-3] + 'xml', cal='ext', qualifier2='d4')  # Get column of sonar files corresponding to light
		for fn in swv_fns :
			sig, fs = sf.read(fn)
			for idx in range(0, len(sig), nsamps):
				ll = sig[idx:idx+nsamps, L_col]
				# Clean the data
				ll = self.fix_light_data(ll)
				# Buffer the data into blocks
				n_blocks = len(ll) // bl
				Y = np.array([ll[i:i+bl] for i in range(0, n_blocks * bl, bl)])
				# Calculate the max of each block and append to L
				L.extend(np.nanmax(Y, axis=1))
		# Ensure L is a column vector
		L = np.array(L)
		if len(L) > 0:
			L = np.append(L, L[-1])  # Add one measurement to equalize length of other sensors
		return L
	
	
	def get_ext_gain(self, path):
		"""
		Get UTC POSIX timestamps and associated gain from xml file
		"""
		tree = ET.parse(path)
		root = tree.getroot()
		times = []
		gainnums = []
		
		# Iterate over all EVENT elements in the XML
		for event in root.findall('EVENT'):
			ext = event.find('EXT')
			if ext is not None:
				gainnum = ext.get('GAINNUM')
				if gainnum is not None:
					gainnums.append(int(gainnum))
					_time = get_ext_time_xml(event.get('TIME'))
					times.append(_time)
		self.gain =  np.array(gainnums)
		self.gain_time = np.array(times)

	def high_pass_filter_light_data(self):
		"""High pass filter the light data to remove low-frequency noise."""
		cutoff = 0.2 / (self.LL['sampling_rate'] / 2)
		self.LL['data_no_filter'] = self.LL['data']
		self.LL['data'] = np.abs(self.fir_nodelay(self.LL['data'], self.LL['sampling_rate'], cutoff, 'high'))


	@staticmethod
	def fir_nodelay(x, n, fp, qual='Hamming'):
		"""
		Delay-free filtering using a linear-phase FIR filter followed by delay correction.
		Parameters:
		x    : np.ndarray
		   The signal to be filtered. It can be multi-channel and 
		   should have a column for each channel.
		n    : int
		   The length of the symmetric FIR filter to use. Must be even.
		fp   : float
		   The filter cut-off frequency relative to the Nyquist frequency (fs/2 = 1).
		qual : str, optional
		   Qualifier to pass to firwin (e.g., window type).
		Returns:
		y    : np.ndarray
		   The filtered signal with delay correction.
		h    : np.ndarray
		   The FIR filter used.
		"""
		n = int(np.floor(n / 2) * 2)  # n must be even for an integer group delay
		noffs = n // 2  # Filter delay
		if qual is not None:
			h = firwin(n+1, fp, pass_zero=qual)
		else:
			h = firwin(n+1, fp)
		if x.ndim == 1:
			x = x[:, np.newaxis]
		# Add padding to the signal
		x_padded = np.vstack([x[n-1::-1, :], x, x[:-n-1:-1, :]])
		# Filter the signal
		y = lfilter(h, 1.0, x_padded, axis=0)
		# Remove padding and correct the delay
		y = y[n+noffs-1:y.shape[0]-n+noffs-1, :]
		return y, h
	    
	
	def fix_light_data(self, L) :
		# Find indices where L > 0.99
		kc = np.where(L > 0.99)[0]
		L[kc] = np.nan
		# Buffer the data into columns of length INTVL to compute median
		n = len(L)
		nbl = int(np.ceil(n / self.INTVL))
		pad_length = (self.INTVL - (n % self.INTVL)) % self.INTVL
		L_padded = np.pad(L, (0, pad_length), constant_values=np.nan)  #To create 2D matrix of size (-1, INTVL)
		Lb = L_padded.reshape(-1, self.INTVL).T
		Lp = np.nanmedian(Lb, axis=1)
		# Subtract the interference pattern from the original signal
		L_corrected = L - np.tile(Lp, nbl)[:n]
		L_corrected = np.append(L_corrected, Lp[:n - len(L_corrected)])
		# Apply a median filter with a kernel size of 3
		L_corrected = medfilt(np.abs(L_corrected), kernel_size=3)
		# Restore the original high values
		L_corrected[kc] = 1 - np.median(Lp)



'''    
	def clean_data(self, ll):
		L = 0
		ll = self.fix_light_sens(ll)
		nbl = int(np.ceil(n / INTVL))

		# Ensure ll is not longer than nsamps
		if len(ll) > nbl:
			ll = ll[:nbl]
		        
		Y = np.array([ll[i:i + bl] for i in range(0, len(ll) - bl + 1, bl)]).T
		        
		 # Calculate the max of each column and append to L
		L.extend(np.max(Y, axis=0))

def get_gain_times(self):
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


'''
