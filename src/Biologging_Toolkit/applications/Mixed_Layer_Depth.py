import numpy as np
from Biologging_Toolkit.processing.Dives import Dives
import netCDF4 as nc

class MixedLayerDepth():
	
	def __init__(self,
			  path : str = None  
			  ):
		self.depth = self.ds['depth'][:].data
		self.temp = self.ds['temperature'][:].data
		self.dive = self.ds['dives'][:].data
	def compute_mld(self):
		up, down = Dives.get_dive_direction(self.depth)
		
	
	
	
