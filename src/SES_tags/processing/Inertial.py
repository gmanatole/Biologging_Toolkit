import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from SES_tags.utils.inertial_utils import *
from SES_tags.utils.format_utils import *
from SES_tags.wrapper import Wrapper
import pdb


class Inertial(Wrapper):
	"""
	Class that uses preprocessed (downsampled and calibrated) accelerometer and magnetometer data to output animal's posture and heading.
	Works on any calibrated data.
	Based on : Benhamou, 2023, Of heading, posture and body rotations derived from data acquired by animal-borne accelerometers, magnetometers and gyrometers, kernel density estimation of the corresponding spherical distributions, and fine-scale movement reconstruction 
	https://doi.org/10.48550/arXiv.2310.05820
	"""
	
	N = None
	
	def __init__(
		self, 
		depid, 
		*, 
		path, 
		inertial_path : str = None,
		data = {'time': None, 'A' : None, 'M' : None, 'P' : None},
		declination : str = None,
		flip : list[list] = [[-1,1,-1], [1,-1,1]],
		ponderation : str = 'angle'
		):
		'''
        Initializes an instance of the Inertial class, which handles loading and processing inertial sensor data.

        Parameters
        ----------
        depid : str
            Identifier corresponding to the individual for whom data is being loaded (e.g., 'ml17_280a').
        path : str
            The path to the main dataset file required by the superclass `Wrapper`.
        inertial_path : str, optional
            Path to the inertial dataset file (e.g., containing magnetometer and accelerometer data). 
            If provided, data will be loaded from this file. Default is None.
		data : dic, optional
			Dictionary containing time data 'time', and inertial data 'A' and 'M'. Depth data 'P' can be added for better euler angle estimations.
	        A : array-like, optional
	            Pre-loaded accelerometer data. Should be provided if `inertial_path` is not given. Default is None.
	        M : array-like, optional
	            Pre-loaded magnetometer data. Should be provided if `inertial_path` is not given. Default is None.
			All data needs to be 1D and correspond to a regularly spaced time data.
        declination : str, optional
            Method to fetch declination data. Options include 'download' or path to declination already downloaded data. Default is 'download'.
			Can be None to not take declination into account
        flip : list of lists, optional
            A matrix used to correct the orientation of the axis to fit the NED (North-East-Down) system.
            The default value is [[-1, 1, -1], [1, -1, 1]], which reflects typical axis inversion adjustments for body frame.
        ponderation : str, optional
            The method of weighting used during heading reconstruction. Options are:
            - 'angle': Uses angular data for weighting.
            - 'speed': Uses speed data for weighting.
            - None.
            Default is 'angle'.

        Raises
        ------
        IndexError
            Raised if the provided dataset does not contain magnetometer data, which is required for processing.
        ValueError
            Raised if neither `inertial_path` nor both `A` and `M` are provided.
        '''
		
		super().__init__(
			depid,
			path
        )
		if inertial_path :
			sens = nc.Dataset(inertial_path)
			depth, self.samplerate, depth_start = sens['P'][:].data, np.round(1/sens['P'].sampling_rate, 2), get_start_time_sens(sens.dephist_device_datetime_start)
			self.inertial_time = np.linspace(0, len(depth), len(depth))*self.samplerate+depth_start    #Create time array for sens data
			self.M, self.A, self.P = sens['M'][:].data, sens['A'][:].data, sens['P'][:].data
		elif data['A'] is not None and data['M'] is not None and data['time'] is not None :
			self.M, self.A, self.P = data['M'], data['A'], data['P']
			self.inertial_time = data['time']
			self.samplerate = self.inertial_time[1]-self.inertial_time[0]

		self._declination = declination   #Fetch declination data in existing dataframe
		self.flip = flip #Correct axis orientation to fit with NED system used by equations in rest of code
		self.ponderation = ponderation    #Ponderation method for heading reconstruction (speed also possible, any other str will lead to no ponderation)

	@property
	def declination(self):
		"""
		Get or set the magnetic declination data for the dataset.
		If the `_declination` attribute is a string, this property behaves as follows:

		- If `_declination` is `'compute'`:
			- Computes the magnetic declination based on the dataset's latitude, longitude, and time data.
			- The computation samples the data at twelve-hour intervals and saved to a CSV file named `'declination.csv'`.
			- `_declination` attribute is updated to point to this file.

		- If `_declination` is a string that points to a CSV file:
			- Loads the declination data from the CSV file.
			- Returns an interpolating function that estimates the declination for any given time in the dataset.

		- If `_declination` is neither `'compute'` nor a valid filename:
			- Defaults to no declination correction.
			- Returns an interpolating function that provides zero correction for all times.

		Returns:
			scipy.interpolate.interp1d: A function that interpolates declination data based on time.
		"""	
		if isinstance(self._declination, str):
			if self._declination == 'download':
				# Case where _declination is 'compute'
				dec_time = self.ds['time'][:].data[:: int(24*3600/self.dt/2)]  # We only need about one declination data every twelve hours
				dec_lat = self.ds['lat'][:].data[:: int(24*3600/self.dt/2)]
				dec_lon = self.ds['lon'][:].data[:: int(24*3600/self.dt/2)]
				dec_data = [get_declination(lat, lon, time) for lat, lon, time in zip(dec_lat, dec_lon, dec_time)]
				pd.DataFrame({'time': dec_time, 'declination': dec_data}).to_csv('declination.csv')
				self.declination = 'declination.csv'
			else:
				# Case where _declination is any other string pointing to a csv file
				dec_data = pd.read_csv(self._declination)
				return interp1d(dec_data.time, dec_data.declination, bounds_error=False, fill_value=None)
		else:
			# Default case: _declination is not a string, no declination correction
			return interp1d(self.ds['time'][:], np.zeros(len(self.ds['time'][:])), bounds_error=False, fill_value=None)


	@declination.setter
	def declination(self, value):
		"""
		Set the `_declination` attribute.

		Parameters:
		----------
		value (str): Can be a string indicating the mode of operation ('compute') or a filename
			     pointing to a CSV file containing precomputed declination data.
		"""
		self._declination = value


	def __call__(self, overwrite = False):
		return self.forward(overwrite = overwrite)
	
	
	def forward(self, overwrite = True):
		"""
		Processes inertial data, normalizes them, computes euler angles and posture, and updates a NetCDF dataset with the results.
		Parameters
		----------
		overwrite : bool
		A flag indicating whether to overwrite existing variables in the NetCDF dataset.
		"""
		
		self.compute_angles()
		
		if overwrite :
			if 'azimuth' in self.ds.variables:
				self.remove_variable('azimuth')
			if 'elevation_angle' in self.ds.variables:
				self.remove_variable('elevation_angle')
			if 'bank_angle' in self.ds.variables:
				self.remove_variable('bank_angle')
			if 'vertical_azimuth' in self.ds.variables:
				self.remove_variable('vertical_azimuth')

		if 'azimuth' not in self.ds.variables:
			azimuth = self.ds.createVariable('azimuth', np.float64, ('time',))
			azimuth.units = 'radians'
			azimuth.long_name = 'Heading of the animal in radians'
			azimuth.measure = 'Angle measured counter-clockwise from the East'
			azimuth.ponderation = 'None'
			azimuth[:] = self.azimuth

		if 'elevation_angle' not in self.ds.variables:
			elevation_angle = self.ds.createVariable('elevation_angle', np.float64, ('time',))
			elevation_angle.units = 'radians'
			elevation_angle.long_name = 'Elevation angle of the animal in radians'
			elevation_angle.measure = 'Angle tail-to-head axis of the animal with regards to the horizontal plane'
			elevation_angle.ponderation = 'None'
			elevation_angle[:] = self.elevation_angle  
			
		if 'bank_angle' not in self.ds.variables:
			bank_angle = self.ds.createVariable('bank_angle', np.float64, ('time',))
			bank_angle.units = 'radians'
			bank_angle.long_name = 'Bank angle of the animal in radians'
			bank_angle.comment = 'Angle representing the roll of the animal'
			bank_angle.measure = 'Angle of the belly-to-back axis of the animal with regards to the vertical plane'
			bank_angle.ponderation = 'None'
			bank_angle[:] = self.bank_angle
		
		if 'vertical_azimuth' not in self.ds.variables:
			vertical_azimuth = self.ds.createVariable('vertical_azimuth', np.float64, ('time',))
			vertical_azimuth.units = 'radians'
			vertical_azimuth.long_name = 'Vertical azimuth angle of the animal in radians'
			vertical_azimuth.measure = 'Angle measured counter-clockwise from the East, relative to the belly-back axis when animal is upright'
			vertical_azimuth.ponderation = 'None'
			vertical_azimuth[:] = self.vertical_azimuth
			
		
	def compute_angles(self) :
		"""
		Compute and average various orientation angles (elevation angle, bank angle, azimuth, and vertical azimuth) 
		over a specified time interval. Additionally, calculates the mean azimuth and rotation of the animal, 
		adjusting for magnetic declination.
		
		Parameters
		----------
		N : float, optional
		The time interval (in seconds) over which to average the inertial data. If not provided, defaults to the class attribute `self.dt`.
		
		This method performs the following steps:
		1. Switches the reference frame to the East-North-UP (ENU) coordinate system.
		2. Averages the inertial data (acceleration, magnetic field, pressure) over the specified 
		interval `N` for each timestep in the reference time.
		3. Computes the local magnetic declination and adjusts the angles accordingly.
		4. Calculates the elevation angle, bank angle, azimuth, and vertical azimuth based on 
		the averaged inertial data.
		5. Computes a weighted mean azimuth over the study period, with optional ponderation based on 
		angle or speed.
		6. Switches the reference frame back to its default state.
		
		Notes
		-----
		- The method assumes that the `self.ds['time']` and `self.inertial_time` arrays are POSIX timestamps epochs.
		"""
		
		if not self.N :
			self.N = self.dt
	
		self.change_axes()   #Switch axis to NED
		self.A_moy = np.full((len(self.ds['time']),3), np.nan)
		self.M_moy = np.full((len(self.ds['time']),3), np.nan)
		self.P_moy = np.full(len(self.ds['time']), np.nan)  
		self.activity = np.full(len(self.ds['time']), np.nan)
		
		if self.ds['time'][:][0] >= self.inertial_time[0] :
			offset = 0
			start_idx = np.argmin(np.abs(self.inertial_time - self.ds['time'][:][0]))
		else :
			offset = len(self.ds['time'][:][self.ds['time'][:] < self.inertial_time[0]])
			start_idx = np.argmin(np.abs(self.inertial_time - self.ds['time'][:][offset]))
		if self.ds['time'] [:][-1] > self.inertial_time[-1] :
			end = len(self.ds['time'][:][self.ds['time'][:] <= self.inertial_time[-1]])
		else :
			end = len(self.ds['time'])
			
		for i, t in tqdm(enumerate(range(offset, end)), position=0, leave=True, desc='Averaging inertial data'):
			lind = int(start_idx + i*(self.dt / self.samplerate))
			hind = lind + int(self.N / self.samplerate) + 1
			self.P_moy[t] = np.nanmean(self.P[lind:hind])
			self.A_moy[t] = list(np.nanmean(self.A[:, lind:hind], axis = 1))
			self.M_moy[t] = list(np.nanmean(self.M[:, lind:hind], axis = 1))
			self.activity[t] = np.sqrt(np.sum(np.nanvar(self.A[:, lind:hind], axis = 1)))
			
		self.A_moy, self.M_moy, self.P_moy, self.activity = np.array(self.A_moy).T, np.array(self.M_moy).T, np.array(self.P_moy), np.array(self.activity)

		#Normalization of acceleration data 
		A_norm = np.sqrt(np.sum(self.A_moy**2, axis = 0))
		self.A_moy = self.A_moy/A_norm		

		#Get local inclination angle for study period and area
		adjust_declination = self.declination(self.ds['time'][:].data) * np.pi / 180

		#Create dP array of correct size by adding slight offset
		self.dP = np.concatenate((np.zeros((1)), self.P_moy[1:] - self.P_moy[:-1]))
		
		# Computing angles 
		self.elevation_angle = np.arcsin(-self.A_moy[0])  #inclinaison longitudinale / elevation angle
		self.bank_angle = np.arctan2(-self.A_moy[1], -self.A_moy[2])    # bank angle
		self.azimuth = np.arctan2((self.A_moy[1]**2 + self.A_moy[2]**2)*self.M_moy[0] - self.A_moy[0]*(self.A_moy[1]*self.M_moy[1] + self.A_moy[2]*self.M_moy[2]), 
							 (self.A_moy[1]*self.M_moy[2] - self.A_moy[2]*self.M_moy[1]))
		self.vertical_azimuth = np.arctan2(self.M_moy[1], -np.sign(self.A_moy[0])*self.M_moy[2]) - np.pi/2   #theta_prime
 
		# Compute average azimuth for length of study period
		if self.ponderation == 'angle':
			ponderation = np.cos(self.elevation_angle)
		elif self.ponderation == 'speed': #HORIZONTAL SPEED PONDERATION
			ponderation = abs(np.concatenate(([0], (self.P_moy[1:]-self.P_moy[:-1])/(self.ds['time'][:][1:]-self.ds['time'][:][:-1]))) / np.tan(self.elevation_angle))
			ponderation[abs(self.elevation_angle) < 5 * np.pi / 180] = np.nan
			ponderation[1:][abs(self.P_moy[1:]-self.P_moy[:-1]) < 0.2] = np.nan
		else :
			ponderation = np.ones((len(self.elevation_angle)))
		self.az_mean = np.arctan2(np.nansum(ponderation*np.sin(self.azimuth)), np.nansum(ponderation*np.cos(self.azimuth))) - adjust_declination
		self.az_mean = modulo_pi(self.az_mean)
		
		# Get orientation of the elephant seal when he is breathing at the surface
		self.rotation = np.zeros((len(self.elevation_angle)))
		self.rotation[self.elevation_angle > 1.5] = np.arctan2(self.M_moy[1][self.elevation_angle > 1.5], self.M_moy[2][self.elevation_angle >= 1.5]) - np.pi/2
		self.rotation[self.elevation_angle < 1.5] = np.arctan2(self.M_moy[1][self.elevation_angle < 1.5], -self.M_moy[0][self.elevation_angle < 1.5]) + self.bank_angle[self.elevation_angle < 1.5]
		self.rotation = self.rotation - adjust_declination
		self.rotation = modulo_pi(self.rotation)
		
		self.change_axes()   #Switch axes back to default


	def change_axes(self):
		self.A[0] = self.flip[0][0] * self.A[0]
		self.A[1] = self.flip[0][1] * self.A[1]
		self.A[2] = self.flip[0][2] * self.A[2]
		self.M[0] = self.flip[1][0] * self.M[0]
		self.M[1] = self.flip[1][1] * self.M[1]
		self.M[2] = self.flip[1][2] * self.M[2]
		
