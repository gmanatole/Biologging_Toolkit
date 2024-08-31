import pandas as pd
import seaborn as sns
from angle_orientation import Orientation
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from SES_tags.wrapper import Wrapper

class Speed(Wrapper) :

	def __init__(self, depid, *, path):
		
		super().__init__(
			depid,
			path
        )

	def _call__(self, overwrite = False) :
		self.forward()
		
	def forward(self, overwrite = False, acoustic = True):
		
		self.from_inertial()
		if acoustic :
			self.from_acoustic()
		
		
	def from_inertial(self):
		"""
		Method to compute speed based on vertical speed and elevation angle.
		Moments where movement is in the horizontal plane are removed.
		"""
		angle = abs(self.ds['elevation_angle'].to_numpy())
		angle[angle < 0.35] = np.nan

		dP = abs(self.ds['depth'][:][1:] - self.ds['depth'][:][:-1])
		dP[dP < 0.5] = np.nan
		
		speed = dP / np.sin(angle[:-1]) / self.ds.dt
		self.inertial_speed = np.append(speed, speed[-1])
		

	def from_acoustic(self, freq = 60):
		pass