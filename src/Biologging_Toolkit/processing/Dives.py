import numpy as np
from Biologging_Toolkit.wrapper import Wrapper
from scipy.signal import find_peaks
import netCDF4 as nc
import os
import pandas as pd
from Biologging_Toolkit.utils.format_utils import *

class Dives(Wrapper): 

	def __init__(self, 
			  depid : str, 
			  *, 
			  path : str, 
			  sens_path : str = None, 
			  raw_path : str = None, 
			  threshold = 20,
			  data = {'time': None, 'P' : None}
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
		
		self.dive_path = os.path.join(path, f'{depid}_dive.csv')
		try :
			self.dive_ds = pd.read_csv(self.dive_path)
		except (FileNotFoundError, pd.errors.EmptyDataError):
			self.dive_ds = pd.DataFrame([])
			self.dive_ds.to_csv(self.dive_path, index = None)
		
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
			
		self.dive_ds['depth'] = np.unique(self.dives)
		begin_time, end_time = [], []
		ref_time = self.ds['time'][:].data
		for dive in np.unique(self.dives) : 
			begin_time.append(np.min(ref_time[self.dives == dive]))
			end_time.append(np.max(ref_time[self.dives == dive]))
		self.dive_ds['begin_time'] = begin_time
		self.dive_ds['end_time'] = end_time
		self.dive_ds.to_csv(self.dive_path, index = None)
		
		
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

	@staticmethod
	def get_bottom_dives(depth, *, prec = 0.10):
		bottom, properties = find_peaks(depth, height = 200)
		for peak in bottom :
			i,j = 0,0
			while (peak+j+2 <= len(depth)) and ((abs(depth[peak] - depth[peak-i]) <= prec*depth[peak]) or (abs(depth[peak] - depth[peak+j]) <= prec*depth[peak])) :
				if (abs(depth[peak] - depth[peak-i]) <= prec*depth[peak]) or (abs(depth[peak] - depth[peak-i-1]) <= prec*depth[peak]):
					bottom = np.append(bottom, peak-i)
					i+=1
				if (abs(depth[peak] - depth[peak+j]) <= prec*depth[peak]) or (abs(depth[peak] - depth[peak+j+1]) <= prec*depth[peak]):
					bottom = np.append(bottom, peak+j)
					j+=1
		return np.unique(bottom)
	
	@staticmethod
	def get_dive_direction(depth, *, remove_bottom = True):
		'''
		Parameters
		----------
		depth : array containing depth data
		remove_bottom : Whether to remove every dive's bottom time, optional. Default to True.
		
		Returns
		-------
		Two arrays containing time and depth when the hydrophone is diving down
		Two arrays containing time and depth data when hydrophone goes back towards the surface
		One mask array where False values are for upward profiles and True values are for downward profiles
		'''
		mask, mask_up, mask_down = np.full((3, len(depth)), True)
		if remove_bottom :
			mask[Dives.get_bottom_dives(depth)] = False
		grad = np.gradient(depth)
		mask_up[grad > 0] = False
		mask_down[grad < 0] = False
		return mask_up & mask, mask_down & mask
