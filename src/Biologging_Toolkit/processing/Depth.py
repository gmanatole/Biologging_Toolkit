import numpy as np
from Biologging_Toolkit.wrapper import Wrapper
from scipy.signal import find_peaks
import netCDF4 as nc
from Biologging_Toolkit.utils.format_utils import *

class Depth(Wrapper): 

	def __init__(self, 
			  depid, 
			  *, 
			  path, 
			  sens_path, 
			  raw_path, 
			  threshold = 20,
			  data = {'time': None, 'A' : None, 'M' : None, 'P' : None}
			  ) :
	
		"""

        Parameters
        ----------
        depid : str
            Identifier corresponding to the individual for whom data is being loaded (e.g., 'ml17_280a').
        path : str
            The path to the main dataset file required by the superclass `Wrapper`.
        sens_data : str, optional
            Path to the sensors dataset file (e.g., containing depth data). 
            If provided, data will be loaded from this file. Default is None.
		data : dic, optional
			Dictionary containing time data 'time', and Depth data 'P'.
	        P : array-like, optional
	            Pre-loaded depth data. Should be provided if `sens_path` is not given. Default is None.
			All data needs to be 1D and correspond to a regularly spaced time data.
		thresold : int
			Depth in meters (> 0) under which a new dive starts. Defaults to 20 m
		"""
		
		super().__init__(
			depid,
			path
        )

		self.raw_path = raw_path
		self.threshold = threshold
		if sens_path :
			sens = nc.Dataset(sens_path)
			depth, self.samplerate, depth_start = sens['P'][:].data, np.round(1/sens['P'].sampling_rate, 2), get_start_time_sens(sens.dephist_device_datetime_start)
			self.sens_time = np.linspace(0, len(depth), len(depth))*self.samplerate+depth_start    #Create time array for sens data
			self.P = sens['P'][:].data
		elif data['P'] is not None and data['time'] is not None :
			self.P =  data['P']
			self.sens_time = data['time']
			self.samplerate = self.inertial_time[1]-self.inertial_time[0]
		
	def __str__(self):
		return "This class identifies surfacing periods and increments dives."

	def __call__(self, overwrite = False):
		self.forward(overwrite)
		
	def forward(self, overwrite = False):
		
		self.dive_count()
		
		if overwrite :
			if 'dives' in self.ds.variables:
				self.remove_variable('dives')

		if 'dives' not in self.ds.variables:
			dives = self.ds.createVariable('dives', np.float64, ('time',))
			dives.long_name = 'Dive number'
			dives.threshold = self.threshold
			dives.threshold_units = 'meters'
			dives.measure = "Increments dives each time the animal goes below threshold"
			dives[:] = self.dives
			
	def get_depth(self):
		self.create_variable('depth', self.P, self.sens_time)

	def dive_count(self) :
		"""
		Counts and increments dives each time the depth gets above specified threshold
		"""
		
		depth = self.ds['depth'][:]
		
		# Set all depth points above threshold to 0 to prevent detections of undesired peaks above thresholds
		surface = np.where(depth < self.threshold)[0]	
		depth[surface] = 0
		peaks, _ = find_peaks(-depth, height = -self.threshold)
		
		#Make dive array
		dives = np.zeros(len(depth))
		for peak in peaks :
			dives[peak :] += 1
		self.dives = dives
