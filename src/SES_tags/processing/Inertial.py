import netCDF4 as nc
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from SES_tags.utils.angular_utils import *
from SES_tags.wrapper import Wrapper

class Inertial(Wrapper):
	
	def __init__(self, ind, *, path, inertial_path = None):
		'''
		Parameters
		----------
		ind : str
			str corresponding to the individual you want to load for (eg. ml17_280a)

		Raises
		------
		IndexError
			One dataset does not have magnetometer data and cannot be used.
		'''
		
		super().__init__(
			ind,
			path
        )
		
		self.A = A
		self.M = M

		self.declinaison()    #Fetch declinaison data in existing dataframe
		self.flip = [[-1,1,-1], [1,-1,1]]     #Correct axis orientation to fit with NED system used by equations in rest of code
		self.ponderation = 'angle'    #Ponderation method for heading reconstruction (speed also possible, anyother str will lead to no ponderation)
		
	@classmethod	
	def from_sens(cls, ind, inertial_path) :
		sens = nc.Dataset(inertial_path)
		depth, depth_dt, depth_start = sens['P'][:].data, np.round(1/sens['P'].sampling_rate, 2), get_start_date(sens.dephist_device_datetime_start)
		time = np.linspace(0, len(depth), len(depth))*depth_dt+depth_start    #Create time array for sens data
		M, A = sens['M'][:].data, sens['A'][:].data
		return cls(ind, A, M)
			
	def declinaison(self) :

		'''dec_data = pd.concat((pd.read_csv('/home6/grosmaan/Documents/codes/declinaison'),
						pd.read_csv('/run/media/grosmaan/LaCie/individus_brut/CTD/declinaison.csv')))
		dec_data = pd.read_csv('/home6/grosmaan/Documents/codes/declinaison')
		dec_data = dec_data[dec_data.datasets == self.ml]
		self.interp = inter((np.unique(dec_data.lon), np.unique(dec_data.lat), np.unique(dec_data.time)), 
					   dec_data.declinaison.to_numpy().reshape(3,3,3), bounds_error = False, fill_value = None)'''
		dec_data = pd.read_csv('/run/media/grosmaan/LaCie/individus_brut/CTD/declinaison.csv')
		dec_data = dec_data[dec_data.datasets == self.ml]
		self.interp = interp1d(dec_data.time, dec_data.declinaison, bounds_error = False, fill_value = None)
		
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
		'''declinaison = self.interp(self.t_moy) * np.pi / 180'''
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

	def calibration(self, t_off = 500, d_off = 0.5, method = 'random', t_min = 0, t_max = np.inf) :
		'''
		For calibration purposes, the aim is to find a dive where the elephant seal travels far in a short amount of time.
		
		Parameters
		----------
		t_off : int, optional
			Minimum time offset between two successive surfacing for dive to be considered. The default is 500.
		d_off : TYPE, optional
			Minimum distance offset between two successive surfacing for dive to be considered. The default is 0.5.
		method : str, optional
			'random', 'all', or 'precise'. 'random' will take a random dive satisfying t_off. 
			'all' will consider every dive between t_min and t_max.
			'precise' will consider dives preselected by user in t_min array.
			The default is 'random'.
		t_min : arr or int or float, optional
			for 'all' method, it is the index for pos_time after which data is kept.
			for 'precise' it is an array of indices from pos_time that are anywhere in a dive. That dive will be kept.
			The default is 0.
		t_max : int or float, optional
			for 'all' method, it is the index for pos_time up to which data is kept.
			The default is np.inf.

		'''
		
		#Select dives for calibration
		pos_dt = self.pos_time[1:]-self.pos_time[:-1]
		dives = np.where((pos_dt > t_off))[0]
		if method == 'random':
			dives = [random.choice(dives)]
		elif method == 'all' :
			dives = dives[(dives > t_min) & (dives < t_max)]
		elif method == 'precise' :
			dives = [len((self.pos_time - _t_min)[(self.pos_time - _t_min) < 0])- 1 for _t_min in t_min]
			
		self.heading = []
		self.heading_time = []
		
		for j, selected_dive in enumerate(dives) :
			pos_up, pos_low = selected_dive+1, selected_dive
			lat = np.array([self.lat[pos_low], self.lat[pos_up]])*np.pi/180
			lon = np.array([self.lon[pos_low], self.lon[pos_up]])*np.pi/180
			#Compute distance traveled by the SES in selected dives and remove if inferior to d_off
			distance = 6371 * np.arccos(coa(lat, lon))
			if distance < d_off :
				continue
			#Compute heading of animal using the lat/lon data
			direction = np.arctan2(np.sin(lat[1]) - coa(lat, lon) * np.sin(lat[0]), np.sin(lon[1] - lon[0]) * np.cos(lat[1]) * np.cos(lat[0]))
			
			#Compute heading of animal using acc/mag
			self.compute_angles(pos_low, pos_up)
			self.heading.append([self.az_mean.mean() * 180 / np.pi, direction * 180 / np.pi])
			self.heading_time.append(self.pos_time[pos_low : pos_up+1].mean())
		
		self.heading = np.array(self.heading)
		self.heading_time = np.array(self.heading_time)
		if len(self.heading) == 0:
			self.heading = np.array([[np.nan, np.nan]])
		
		self.errs = ((self.heading[:,0]-self.heading[:,1]) + 180) % 360 - 180
		self.corr1 = np.sqrt((np.nansum(np.cos(self.heading[:,0] - self.heading[:,1]))**2 + np.nansum(np.sin(self.heading[:,0] - self.heading[:,1]))**2))/self.heading.shape[0]
		self.corr2 = np.sqrt((np.nansum(np.cos(self.heading[:,0] + self.heading[:,1]))**2 + np.nansum(np.sin(self.heading[:,0] + self.heading[:,1]))**2))/self.heading.shape[0]
	
	def change_axes(self):
		self.A[0] = self.flip[0][0] * self.A[0]
		self.A[1] = self.flip[0][1] * self.A[1]
		self.A[2] = self.flip[0][2] * self.A[2]
		self.M[0] = self.flip[1][0] * self.M[0]
		self.M[1] = self.flip[1][1] * self.M[1]
		self.M[2] = self.flip[1][2] * self.M[2]
		

'''mls = ['ml21_305a', 'ml19_294b', 'ml19_294a', 'ml19_293b', 'ml19_292b', 'ml19_290c', 'ml19_290b', 'ml19_290a', 'ml18_296a', 'ml18_295b', 'ml18_295a', 'ml18_294c', 'ml17_280a', 'ml17_281a', 'ml17_301a', 'ml18_292a', 'ml18_293a', 'ml18_294a', 'ml18_294b', 'ml19_292a', 'ml19_293a', 'ml19_295a', 'ml19_295c', 'ml19_296a','ml20_292a', 'ml20_293a', 'ml20_295a', 'ml20_296a','ml20_296b', 'ml20_296c','ml20_297a','ml20_304a','ml20_313a','ml21_282a','ml21_295a','ml21_297a','ml21_298a','ml21_303a', 'ml21_305b']
for ml in mls:
    inst = Orientation(ml)
    inst.compute_angles()
    df = pd.DataFrame().from_dict({'epoch':inst.t_moy, 'depth':inst.P_moy, 'Ax':inst.A_moy[0], 'Ay':inst.A_moy[1], 'Az':inst.A_moy[2], 'Mx':inst.M_moy[0], 'My':inst.M_moy[1], 'Mz':inst.M_moy[2], 'elevation_angle':inst.elevation_angle, 'bank_angle':inst.bank_angle, 'azimuth':inst.azimuth, 'vertical_azimuth':inst.vertical_azimuth, 'activity':inst.activity, 'rotation':inst.rotation})
    df.to_csv('/run/media/grosmaan/LaCie/individus_brut/CTD/'+ml+'/inertial_data.csv', index = None)'''
