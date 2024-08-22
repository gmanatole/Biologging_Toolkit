import netCDF4 as nc
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from SES_tags.utils.angular_utils import *
from SES_tags.wrapper import Wrapper



class Inertial(Wrapper):
	
	def __init__(
		self, 
		ind, 
		*, 
		path, 
		inertial_path : str = None,
		A = None,
		M = None,
		declination : str = 'download',
		flip : list[list] = [[-1,1,-1], [1,-1,1]],
		ponderation : str = 'angle'
		):
		'''
        Initializes an instance of the Inertial class, which handles loading and processing inertial sensor data.

        Parameters
        ----------
        ind : str
            Identifier corresponding to the individual for whom data is being loaded (e.g., 'ml17_280a').
        path : str
            The path to the main dataset file required by the superclass `Wrapper`.
        inertial_path : str, optional
            Path to the inertial dataset file (e.g., containing magnetometer and accelerometer data). 
            If provided, data will be loaded from this file. Default is None.
        A : array-like, optional
            Pre-loaded accelerometer data. Should be provided if `inertial_path` is not given. Default is None.
        M : array-like, optional
            Pre-loaded magnetometer data. Should be provided if `inertial_path` is not given. Default is None.
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
			ind,
			path
        )
		if inertial_path :
			sens = nc.Dataset(inertial_path)
			depth, depth_dt, depth_start = sens['P'][:].data, np.round(1/sens['P'].sampling_rate, 2), get_start_date(sens.dephist_device_datetime_start)
			self.inertial_time = np.linspace(0, len(depth), len(depth))*depth_dt+depth_start    #Create time array for sens data
			self.M, self.A = sens['M'][:].data, sens['A'][:].data
		elif A & M :
			self.M, self.A = M, A

		self._declination = declination   #Fetch declination data in existing dataframe
		self.flip = flip #Correct axis orientation to fit with NED system used by equations in rest of code
		self.ponderation = ponderation    #Ponderation method for heading reconstruction (speed also possible, any other str will lead to no ponderation)

	@property
	def declination(self):
		if isinstance(self._declination, str):
			match self._declination:
				case 'compute':
					dec_time = self.ds['time'][:].data[:: int(24*3600/self.dt/2)]  #We only need about one declination data every twelve hours
					dec_lat = self.ds['lat'][:].data[:: int(24*3600/self.dt/2)] 
					dec_lon = self.ds['lon'][:].data[:: int(24*3600/self.dt/2)]
					dec_data = [get_declination(lat, lon, time) for lat, lon, time in zip(dec_lat, dec_lon, dec_time)]
					pd.DataFrame({'time':dec_time, 'declination':dec_data}).to_csv('declination.csv')
					self.declination = 'declination.csv'
				case _:
					dec_data = pd.read_csv(self._declination)
					return interp1d(dec_data.time, dec_data.declination, bounds_error=False, fill_value=None)
				else:
					# Default to no declination correction
					return interp1d(self.time, np.zeros(len(self.time)), bounds_error=False, fill_value=None)

	@declination.setter
	def declination(self, value):
		self._declination = value

		
	def compute_angles(self, t_min = 0, t_max = -1, N = 3) :
		'''
		Parameters
		----------
		t_min : int, optional
			Index for trk time array at which study starts. 
			For instance if user wants to look at data after epoch 15785896, t_min will verify pos_time[t_min] = 15785896
			The default is 0.
		t_max : TYPE, optional
			Index for trk time array at which study stops. 
			For instance if user wants to look at data before epoch 15785907, t_min will verify pos_time[t_min] = 15785907
			The default is -1
		N : int, optional
			Window used to average acc and mag data to remove components due to animal's movement. The default is 3.
		'''
	
		self.change_axes()   #Switch axis to NED
		if t_max == -1:    #If user wants to compute angles for all available data
			t_max = len(self.pos_time)-1
		# Create new arrays for user's time selection
		pos_bounds = self.pos_time[t_min : t_max+1]
		lat_sel, lon_sel = self.lat[t_min : t_max+1], self.lon[t_min : t_max+1]
		lat_interp, lon_interp = interp1d(pos_bounds, lat_sel), interp1d(pos_bounds, lon_sel)
		t_etude = self.time[(self.time > pos_bounds[0]) & (self.time < pos_bounds[-1])]
		A_etude = self.A[:, (self.time > pos_bounds[0]) & (self.time < pos_bounds[-1])]
		M_etude = self.M[:, (self.time > pos_bounds[0]) & (self.time < pos_bounds[-1])]
		P_etude = self.depth[(self.time > pos_bounds[0]) & (self.time < pos_bounds[-1])]

		# Remove noise due to elephant seal's movement
		A_moy, M_moy, t_moy, P_moy, activity = [],[], [], [], []
		for i in tqdm(range(0, A_etude.shape[1], int(N/self.depth_dt))):   #Averaging over N seconds
			P_moy.append(np.nanmean(P_etude[i:i+int(N/self.depth_dt)+1]))
			t_moy.append(np.nanmean(t_etude[i:i+int(N/self.depth_dt)+1]))
			A_moy.append(list(np.nanmean(A_etude[:, i:i+int(N/self.depth_dt)+1], axis = 1)))
			M_moy.append(list(np.nanmean(M_etude[:, i:i+int(N/self.depth_dt)+1], axis = 1)))
			activity.append(np.sqrt(np.sum(np.nanvar(A_etude[:, i:i+int(N/self.depth_dt)+1], axis = 1))))
			
		self.A_moy, self.M_moy, self.P_moy, self.t_moy, self.activity = np.array(A_moy).T, np.array(M_moy).T, np.array(P_moy), np.array(t_moy), np.array(activity)

		#Normalization of acceleration data 
		A_norm = np.sqrt(np.sum(self.A_moy**2, axis = 0))
		self.A_moy = self.A_moy/A_norm		

		#Get local inclination angle for study period and area
		declinaison = self.interp((lon_interp(self.t_moy), lat_interp(self.t_moy), self.t_moy)) * np.pi / 180

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
			ponderation = abs(np.concatenate(([0], (self.P_moy[1:]-self.P_moy[:-1])/(self.t_moy[1:]-self.t_moy[:-1]))) / np.tan(self.elevation_angle))
			ponderation[abs(self.elevation_angle) < 5 * np.pi / 180] = np.nan
			ponderation[1:][abs(self.P_moy[1:]-self.P_moy[:-1]) < 0.2] = np.nan
		else :
			ponderation = np.ones((len(self.elevation_angle)))
		self.az_mean = np.arctan2(np.nansum(ponderation*np.sin(self.azimuth)), np.nansum(ponderation*np.cos(self.azimuth))) - declinaison
		self.az_mean = modulo_pi(self.az_mean)
		
		# Get orientation of the elephant seal when he is breathing at the surface
		self.rotation = np.zeros((len(self.elevation_angle)))
		self.rotation[self.elevation_angle > 1.5] = np.arctan2(self.M_moy[1][self.elevation_angle > 1.5], self.M_moy[2][self.elevation_angle >= 1.5]) - np.pi/2
		self.rotation[self.elevation_angle < 1.5] = np.arctan2(self.M_moy[1][self.elevation_angle < 1.5], -self.M_moy[0][self.elevation_angle < 1.5]) + self.bank_angle[self.elevation_angle < 1.5]
		self.rotation = self.rotation - declinaison
		self.rotation = modulo_pi(self.rotation)
		
		self.change_axes()   #Switch axes back to default


	def change_axes(self):
		self.A[0] = self.flip[0][0] * self.A[0]
		self.A[1] = self.flip[0][1] * self.A[1]
		self.A[2] = self.flip[0][2] * self.A[2]
		self.M[0] = self.flip[1][0] * self.M[0]
		self.M[1] = self.flip[1][1] * self.M[1]
		self.M[2] = self.flip[1][2] * self.M[2]
		
