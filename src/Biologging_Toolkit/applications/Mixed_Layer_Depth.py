import numpy as np
from Biologging_Toolkit.processing.Dives import Dives
from Biologging_Toolkit.utils.format_utils import *
import netCDF4 as nc
from tqdm import tqdm 

class MixedLayerDepth(Dives) :
	
	def __init__(self,
			  depid : str,
			  *,
			  path : str = None  
			  ):
		
				
		super().__init__(
			depid,
			path = path
        )
		
		self.depth = self.ds['depth'][:].data
		self.temp = self.ds['temperature'][:].data
		self.dives = self.ds['dives'][:].data
		self.criterion = 0.03
		
	@staticmethod
	def profile_check(depth, profile, sr) :
		if len(profile) >= 3 * 60 / sr :
			indices = np.argsort(depth)
			depth = depth[indices]
			profile = profile[indices]
			return depth[depth > 5], profile[depth > 5]
		else :
			return [], []
		
	def compute_mld(self):
		up, down = self.get_dive_direction(self.depth[:: 60 // self.ds.sampling_rate])
		up, down = resample_boolean_array(up, len(self.depth)), resample_boolean_array(down, len(self.depth))
		self.mld, self.mld_estimation =[], []
		for dive in np.unique(self.dives) :
			_mld, _mld_est = [], []
			for direction in [up, down]:
				_temp = self.temp[(self.dives == dive) & direction]
				_depth = self.depth[(self.dives == dive) & direction]
				_depth, _temp = self.profile_check(_depth, _temp, self.ds.sampling_rate)
				delta_k, std_k, X_k = [],[],[]
				for k in range(len(_temp)):
					delta_k.append(np.sqrt( 1/(k+1) * np.nansum([((_temp[i] - np.nanmean(_temp[:k+1]))**2) for i in range(k)])))
					std_k.append(np.nanmax(_temp[:k+1]) - np.nanmin(_temp[:k+1]))
					X_k.append(delta_k[-1] / std_k[-1])
				pos_zn1 = np.nanargmin(X_k)
				_mld_est.append(_depth[pos_zn1])
				for step in range(len(delta_k[:pos_zn1 - 1]), -1, -1):
					if (delta_k[step + 1] - delta_k[step]) / (delta_k[pos_zn1 + 1] - delta_k[pos_zn1]) <= self.criterion :
						_mld.append(_depth[step])
						break
			self.mld.append(_mld)
			self.mld_estimation.append(_mld_est)
		
	
	
	
