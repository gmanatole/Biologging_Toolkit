import numpy as np
from scipy.optimize import curve_fit, Bounds
from Biologging_Toolkit.wrapper import Wrapper
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


class Wind_Direction(Wrapper) :

	def __init__(self, 
			  depid, 
			  *, 
			  path, 
			  ):
		
		"""
		This class uses processed dataset to estimate the animal's speed.
		The main method is to use Euler angles to get the speed from the pitch and vertical speed.
		If acoustic data is available in the data structure a model can be fitted using the previous speed estimation.
		"""
		
		super().__init__(
			depid,
			path
        )


	def __call__(self, acoustic = True, overwrite = False) :
		self.forward(acoustic, overwrite)
		
	def forward(self, acoustic = True, overwrite = False):
		
		self.from_inertial()
		if overwrite :
			if 'inertial_speed' in self.ds.variables:
				self.remove_variable('inertial_speed')

		if 'inertial_speed' not in self.ds.variables:
			inertial = self.ds.createVariable('inertial_speed', np.float64, ('time',))
			inertial.units = 'm/s'
			inertial.long_name = 'Speed of the animal'
			inertial.measure = 'Computed using the elevation angle and the vertical speed'
			inertial.note = 'Dive portions where elephant seal has elevation angle below 0.35 rad or vertical speed below 0.5 m/s are removed'
			inertial[:] = self.inertial_speed
		
		if acoustic :
			self.from_acoustic()
			if overwrite :
				if 'acoustic_speed' in self.ds.variables:
					self.remove_variable('acoustic_speed')
	
			if 'acoustic_speed' not in self.ds.variables:
				acoustic = self.ds.createVariable('acoustic_speed', np.float64, ('time',))
				acoustic.units = 'm/s'
				acoustic.long_name = 'Speed of the animal'
				acoustic.measure = f'Computed using flow noise at {self.freq} and fitted on inertial speed'
				acoustic[:] = self.acoustic_speed
				
	def get_average_posture(self):

	    #Get azimuth as a function of elevation angle
	    orientation = inst.azimuth
	    orientation[inst.elevation_angle > 1.40] = inst.vertical_azimuth 
	    
	    #Get surface times with dives longer than t_off
	    time_data = inst.epoch[inst.depth < 1].to_numpy()
	    pos_dt = time_data[1:]-time_data[:-1]
	    dives = np.where((pos_dt > t_off))[0]
	    upper_bound, lower_bound = time_data[dives][1:], time_data[dives+1][:-1]
	    avg_rot, avg_time, _rot, _time, len_rot = [], [], [], [], []
	    
	    plt.figure(figsize=(12,7))
	
	    #Get orientation data for those dives (remove beginning and end of surfacing)
	    for time_up, time_down in zip(upper_bound, lower_bound) :
	        len_rot.append(abs(time_up - time_down))
	        rot_data = modulo_pi(orientation[(inst.epoch < time_up) & (inst.epoch > time_down)] + np.pi/2)
	        if graph :
	            plt.scatter(inst.epoch[(inst.epoch < time_up) & (inst.epoch > time_down)], 
	                        rot_data, color='gold',s=2, alpha = 0.2)
	        _rot.append(angular_average(rot_data[5:-5]))
	        _time.append(np.nanmean([time_up, time_down]))
	    #Rolling average of surface orientations
	    orientation_conv = pd.Series(_rot).rolling(window = 15, min_periods = 5, center = True).median().to_numpy()
	    
	    if graph :
	        cmap = cm.get_cmap('Reds')
	        plt.scatter(aux.time, modulo_pi(np.arctan2(aux.v10, aux.u10)), c = np.sqrt(aux.u10**2 + aux.v10**2), 
	                    s = 5, cmap = cmap, label='wind direction')
	        '''plt.scatter(aux.time, modulo_pi(aux.mdts*np.pi/180+np.pi/2), label='Direction swell',s=2)
	        plt.scatter(aux.time, modulo_pi(aux.mdww*np.pi/180+np.pi/2), label='Direction wind waves',s=2)
	        plt.scatter(aux.time, modulo_pi(aux.mwd*np.pi/180+np.pi/2), label='Direction wave',s=2)
	        plt.scatter(aux.time, modulo_pi(aux.dwi*np.pi/180+np.pi/2), label='Direction wave',s=2)'''
	        plt.plot(_time, orientation_conv, label = 'Averaged posture')
	        plt.grid(axis = 'y')
	        plt.xticks(ticks = aux.time[::len(aux.time)//10], labels = [datetime.fromtimestamp(ts) for ts in aux.time[::len(aux.time)//10]], rotation = 70)
	        plt.ylabel('Wind direction and Body orientation wrt to the East (rad)')
	        # Creating custom legend handles
	        gold_scatter = mpatches.Patch(color='gold', label='all posture data (rad)')
	        blue_line = mlines.Line2D([], [], color='blue', label='averaged posture data (rad)')
	        red_line = mlines.Line2D([], [], color='red', label='wind direction (rad)')
	        plt.legend(handles=[gold_scatter, blue_line, red_line])
	
	        plt.colorbar()
	        plt.show()
	        
	    if corr :
	        #Correlation
	        wind_orientation = interp(aux.time, np.arctan2(aux.v10, aux.u10), bounds_error = False)(_time)
	        y1 = interp(aux.time, modulo_pi(aux.mdts*np.pi/180+np.pi/2), bounds_error = False)(_time)
	        y2 = interp(aux.time, modulo_pi(aux.mdww*np.pi/180+np.pi/2), bounds_error = False)(_time)
	        y3 = interp(aux.time, modulo_pi(aux.mwd*np.pi/180+np.pi/2), bounds_error = False)(_time)
	        y4 = interp(aux.time, modulo_pi(aux.dwi*np.pi/180+np.pi/2), bounds_error = False)(_time)
	        corr = []
	
	        for env_var in (wind_orientation,y1,y2,y3,y4):
	            corr.append(angular_correlation(orientation_conv, env_var))
	        corr_df = pd.DataFrame(corr, columns = ['positive','negative'])
	        corr_df['var'] = ['wind direction', 'mdts', 'mdww', 'mwd', 'dwi']
	        return corr_df
	    
	    if plot_errors :
	        cmap = cm.get_cmap('Reds')
	        az = modulo_pi(interp(aux.time, aux.sun_azimuth, bounds_error = False)(_time))
	        zen = interp(aux.time, aux.sun_zenith, bounds_error = False)(_time)
	        wind_dir = modulo_pi(interp(aux.time, np.arctan2(aux.v10, aux.u10), bounds_error=False)(_time))
	        y1 = interp(aux.time, np.sqrt(aux.u10**2 + aux.v10**2), bounds_error = False)(_time)
	        y2 = interp(aux.time, modulo_pi(aux.mdww*np.pi/180+np.pi/2), bounds_error = False)(_time)
	        y3 = interp(aux.time, modulo_pi(aux.mwd*np.pi/180+np.pi/2), bounds_error = False)(_time)
	        y4 = interp(aux.time, modulo_pi(aux.dwi*np.pi/180+np.pi/2), bounds_error = False)(_time)
	
	        azimuth = abs(modulo_pi(az+np.pi-wind_dir))
	        
	        err = abs(modulo_pi(wind_dir - orientation_conv))
	        fig, ax = plt.subplots(1,3, figsize = (15, 7))
	        ax = ax.flatten()
	        ax[0].scatter(azimuth, err)
	        ax[1].scatter(zen, err)
	        ax[2].scatter(y1, err)
	
	    plt.show()
    
