import numpy as np
from SES_tags.wrapper import Wrapper
from scipy.signal import find_peaks

class Depth(Wrapper): 

	def __init__(self, depid, *, path, raw_path, threshold = 20) :
		
		"""
		Parameters
		----------
		thresold : int
		Depth in meters (> 0) under which a new dive starts. Defaults to 20 m
		"""
		
		super().__init__(
			depid,
			path
        )
		
		self.raw_path = raw_path
		self.threshold 
		
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
			dives[:] = self.dives
			
	def get_depth(self):
		pass

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