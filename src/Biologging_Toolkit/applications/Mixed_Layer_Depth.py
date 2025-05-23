import numpy as np
from Biologging_Toolkit.processing.Dives import Dives
from Biologging_Toolkit.utils.format_utils import *
from tqdm import tqdm 
import pandas as pd
import netCDF4 as nc
import gsw
from datetime import datetime, timedelta, timezone


class MixedLayerDepth(Dives) :
	
	def __init__(self,
			  depid : str,
			  *,
			  path : str = None,
			  meop_path : str = None,
			  threshold_temperature = 0.2,  #0.2 from Dong, 2008
			  threshold_density = 0.03,   #Montegut 2004
			  criterion = 0.3
			  ):
		
				
		super().__init__(
			depid,
			path = path
        )
		
		self.meop_path = meop_path
		try :
			self.depth = self.ds['depth'][:].data
			self.temp = self.ds['temperature'][:].data
			self.dives = self.ds['dives'][:].data
		except IndexError:
			print('Depth, temperature or dive data is not found in reference structure')
		self.criterion = criterion
		self.threshold_temperature = threshold_temperature
		self.threshold_density = threshold_density


	def __call__(self, overwrite = False):
		return self.forward(overwrite = overwrite)
	
	def forward(self, overwrite = True, method = 'meop', variable = 'sigma0'):
		
		if method == 'raw_threshold':
			self.get_threshold_mld()
			self.threhsold_mld = np.array(self.threshold_mld).reshape(-1,2)
			self.dive_ds['threhsold_up'] = self.threhsold_mld[:,0]
			self.dive_ds['threhsold_down'] = self.threhsold_mld[:,1]
		if method == 'raw_variance' :
			self.get_relative_variance_mld()
			self.dive_ds['zn1_up'] = np.array(self.zn1)[:,1]
			self.dive_ds['zn2_up'] = np.array(self.zn2)[:,1]
			self.dive_ds['zn1_down'] = np.array(self.zn1)[:,0]
			self.dive_ds['zn2_down'] = np.array(self.zn2)[:,0]
		if method == 'meop' :
			threshold = self.threshold_density if variable == 'sigma0' else self.threshold_temperature
			time_mld, mld, gradient, density, temp10 = self.ctd_mld(self.meop_path, variable, threshold)
			indices_mld = np.searchsorted(self.dive_ds.begin_time, time_mld[time_mld < self.dive_ds.end_time.iloc[-1]])
			final_mld = np.full(len(self.dive_ds), np.nan)
			final_mld[indices_mld-1] = mld[time_mld < self.dive_ds.end_time.iloc[-1]]
			self.dive_ds['mld'] = final_mld
		self.dive_ds.to_csv(self.dive_path, index = None)
		
	'''
	def forward(self, window_size = 10, overwrite = True):
		self.compute_mld()
		zn1_corr, zn2_corr = [], []
		for i in range(len(self.zn1)):
		    if self.zn1[i] != 0:
		        zn1_corr.append(np.nanmean(self.zn1[i]))
		    else :
		        zn1_corr.append(np.nan)
		    if self.zn2[i] != 0:
		        zn2_corr.append(np.nanmean(self.zn2[i]))
		    else :
		        zn2_corr.append(np.nan)
		zn1_series = pd.Series(zn1_corr)
		self.zn1_smoothed = zn1_series.rolling(window=window_size, min_periods=1).mean().to_numpy()
		zn2_series = pd.Series(zn2_corr)
		self.zn2_smoothed = zn2_series.rolling(window=window_size, min_periods=1).mean().to_numpy()
		self.dive_ds['zn1'] = self.zn1_smoothed
		self.dive_ds['zn2'] = self.zn2_smoothed
		self.dive_ds.to_csv(self.dive_path, index = None)'''
	
	@staticmethod
	def ctd_mld(ctd_path, variable = 'sigma0', threshold = 0.03):

		ctd_ds = nc.Dataset(ctd_path)
		ctd_time = np.array([(datetime(1950,1,1,0,0,0) + timedelta(elem)).replace(tzinfo=timezone.utc).timestamp() for elem in ctd_ds['JULD'][:].data])
		if np.all(ctd_ds['PSAL_ADJUSTED'][:].mask):
			sal_var = 'PSAL'
		else :
			sal_var = 'PSAL_ADJUSTED'
		if np.all(ctd_ds['TEMP_ADJUSTED'][:].mask):
			temp_var = 'TEMP'
		else :
			temp_var = 'TEMP_ADJUSTED'
		temp = ctd_ds[temp_var][:].data
		temp[ctd_ds[temp_var][:].mask] = np.nan
		sal = ctd_ds[sal_var][:].data
		sal[ctd_ds[sal_var][:].mask] = np.nan
		sigma0 = gsw.density.sigma0(sal, temp)
		ctd_mld, gradient_value, density_value, temp10 = [], [], [], []
		if variable == 'sigma0' :
			for i, profile in enumerate(sigma0):
				try :
					ctd_mld.append(np.min(np.where(abs(profile[11:] - profile[10]) > threshold))+11)
					gradient_value.append(abs(profile[ctd_mld[-1]] - profile[ctd_mld[-1]-1]))
					density_value.append(profile[ctd_mld[-1]])
					temp10.append(temp[i,10])
				except ValueError :
					ctd_mld.append(np.nan)
					gradient_value.append(np.nan)
					density_value.append(np.nan)
					temp10.append(np.nan)
		if variable == 'temperature' :
			for profile in temp:
				try :
					ctd_mld.append(np.min(np.where(abs(profile[11:] - profile[10]) > threshold))+11)
					gradient_value.append(abs(profile[ctd_mld[-1]] - profile[10]))
					temp10.append(profile[10])
				except ValueError :
					ctd_mld.append(np.nan)
					gradient_value.append(np.nan)
					temp10.append(np.nan)
		return ctd_time, np.array(ctd_mld), np.array(gradient_value), np.array(density_value), np.array(temp10)

	@staticmethod
	def profile_check(depth, profile) :
		indices = np.argsort(depth)
		depth = depth[indices]
		profile = profile[indices][depth > 10]
		depth = depth[depth > 10]
		if len(depth) == 0 :
			return [], []
		if np.max(depth) < 50 :
			return [], []
		if np.min(depth)  > 20 :
			return [], []
		if abs(np.mean(np.diff(depth))) > 6 :  #Changed from dz = 2 to dz = 6 as most dz for SES are below 5 (criteria for consistency of profiles)
			return [], []
		if len(depth[(depth >= 10) & (depth <= 40)]) < 2 :
			return [], []
		if np.max(profile[depth <= 20]) - np.min(profile[depth <= 20]) >= np.max(profile[depth >= 20]) - np.min(profile[depth >= 20]) :
			return [], []
		if np.max(profile) - np.min(profile) < 1 :
			return [], []
		return depth, profile
	
	@staticmethod
	def threshold_profile_check(depth, profile) :
		indices = np.argsort(depth)
		depth = depth[indices]
		profile = profile[indices]
		return depth[depth > 10], profile[depth > 10]  #Depth to keep from Montegut, 2004 is 10 m whereas Dong, 2008 uses 20m

	
	def get_threshold_mld(self) :
		if self.up is None or self.down is None :
			self.dive_profile()
			
		self.threshold_mld = []
		unique_dives = np.unique(self.dives)
		#Look at each dive individually
		for dive in tqdm(unique_dives, desc = 'Computing threshold MLD for each dive') :
			# Get estimations from two profiles : up and down trajectories
			for direction in [self.up, self.down]:		
				_temp = self.temp[(self.dives == dive) & direction]
				_depth = self.depth[(self.dives == dive) & direction]
				_depth, _temp = self.threshold_profile_check(_depth, _temp)

				i = 0
				while (i < len(_temp)) and (abs(_temp[i] - _temp[0]) < self.threshold_temperature):
					i+=1
				if i == len(_temp) or abs(_temp[i] - _temp[0]) > 0.7 :  #Flag criterion from Montegut, 2004
					self.threshold_mld.append(np.nan)
				else :
					self.threshold_mld.append(_depth[i])
				
	def get_relative_variance_mld(self):
		
		if self.up == None or self.down == None :
			self.dive_profile()
			
		self.zn2, self.zn1 =[], []
		unique_dives = np.unique(self.dives)
		#Look at each dive individually
		for dive in tqdm(unique_dives, desc = 'Computing MLD for each dive') :
			_zn1, _zn2 = [], []
			# Get estimations from two profiles : up and down trajectories
			for direction in [self.up, self.down]:
				_temp = self.temp[(self.dives == dive) & direction]
				_depth = self.depth[(self.dives == dive) & direction]
				#Sort temp and depth profile by increasing depth and check if there is enough data
				_depth, _temp = self.profile_check(_depth, _temp)
				delta_k, std_k, X_k = [],[],[]
				for k in range(len(_temp)):
					delta_k.append(np.sqrt( 1/(k+1) * np.nansum([((_temp[i] - np.nanmean(_temp[:k+1]))**2) for i in range(k+1)])))
					std_k.append(np.nanmax(_temp[:k+1]) - np.nanmin(_temp[:k+1]))
					X_k.append(delta_k[-1] / std_k[-1])
				#If X_k contains only nan values, add nan
				try : 
					pos_zn1 = np.nanmin((np.nanargmin(X_k), len(X_k)-2))  #If lowest X_k is last element, take penultimate element
					_zn1.append(_depth[pos_zn1])
				except ValueError :
					_zn1.append(np.nan)
					_zn2.append(np.nan)
					continue
				_zn2_temp = np.nan
				for step in range(len(delta_k[:pos_zn1 - 1]), -1, -1):
					if (delta_k[step + 1] - delta_k[step]) / (delta_k[pos_zn1 + 1] - delta_k[pos_zn1]) <= self.criterion :
						_zn2_temp = _depth[step]
						break
				_zn2.append(_zn2_temp)
			self.zn_time.append(inst.dive_ds)
			self.zn2.append(_zn2)
			self.zn1.append(_zn1)


	def get_wind_correlation(self) :
		for i in range(24):
			df[f'wind_{i}h'] = np.nan
			for depid in depids :
				wind_interp = interp1d(df[df.ses == depid].end_time, df[df.ses == depid].wind_speed, bounds_error = False)
				df[f'wind_{i}h'][df.ses == depid] = wind_interp(df[df.ses == depid].end_time - i*3600)
		
'''import numpy as np
import xarray as xr
import cf_xarray

def compute_mld(ds, grid, density_threshold=0.02,temp_threshold=0.2):

    """
    Compute the mixed layer depth with a density threshold criteria on sigma0.
    If the mld is equal to the maximum depth of the profile, it is replaced by a NaN.

    Parameters
    ----------
    ds : xarray.Dataset
        ds must contain sigma0 with the cf compliant 'standard_name' attribute:
        {'standard_name':'sea_water_sigma_t'}
        The depth dimension must have the {'axis':'Z'} attribute.
    grid : xgcm.Grid
        Grid associated with ds.
    density_threshold : float, default 0.02
       Threshold applied on density to determine the MLD

    Returns
    -------
    xarray.DataArray
    """

    out = xr.Dataset()
    # We don't use the xarray interpolation due to the presence of NaNs,
    # it is written in the scipy.interpolate.interp1d that data should not have NaNs
    # out['sigma0_10_m'] = ds.sigma0.where(~np.isnan(ds.sigma0), drop=True).interp(depth=10).drop_vars(['depth'], errors='ignore')

    #depth = ds.cf.axes["Z"][0]
   
    # Get sigma0 at 10 m depth

    if np.isnan(ds.cf['sea_water_sigma_t']).all()==False:
        out["sigma0_10_m"] = grid.transform(
            ds.cf['sea_water_sigma_t'], "Z", np.array([10]), target_data=ds.cf['depth']
        ).squeeze()
        out["sigma0_diff_from_surf"] = xr.where(
            ds.cf['depth'] >= 10, ds.cf['sea_water_sigma_t'] - out.sigma0_10_m, 0
        )

        out['depth'] = ds.cf['depth']

        out["mld"] = (
            grid.transform(
                out['depth'].broadcast_like(out.sigma0_diff_from_surf),
                "Z",
                np.array([density_threshold]),
                target_data=out.sigma0_diff_from_surf,
            )
            .isel(sigma0_diff_from_surf=0)
            .drop_vars(["sigma0_diff_from_surf"])
        )

        out["mld"].attrs.update({"MLD_calculation":"temperature and salinity available"})

 

    else:
        out["temp_10_m"] = grid.transform(
            ds.cf['sea_water_potential_temperature'], "Z", np.array([10]), target_data=ds.cf['depth']
        ).squeeze()

        out["temp_diff_from_surf"] = xr.where(
            ds.cf['depth'] >= 10, ds.cf['sea_water_potential_temperature'] - out.temp_10_m, 0
        )

        out['depth'] = ds.cf['depth']

        out["mld"] = (
            grid.transform(
                out['depth'].broadcast_like(out.temp_diff_from_surf),
                "Z",
                np.array([temp_threshold]),
                target_data=out.temp_diff_from_surf,
            )
            .isel(temp_diff_from_surf=0)
            .drop_vars(["temp_diff_from_surf"])
        )

        out["mld"].attrs.update({"MLD_calculation":"only temperature available"})
		         
    # remove mld where mld >= max(depth)
    out["mld"] = out["mld"].where(out["mld"] < out['depth'].cf.max("Z"))

    # cf attribute
    out["mld"].attrs.update({"standard_name": "ocean_mixed_layer_thickness"})

    # see https://github.com/xarray-contrib/cf-xarray/issues/285
    try:
        out["mld"] = out["mld"].cf.add_canonical_attributes()

    except IndexError:
        pass
    return out["mld"]




import matplotlib.pyplot as plt
def plot(inst, aux, window_size = 10):
	mld_corr = []
	for i in range(len(inst.mld)):
	    if inst.mld_estimation[i] != 0:
	        mld_corr.append(np.nanmean(inst.mld_estimation[i]))
	    else :
	        mld_corr.append(np.nan)		
	signal_series = pd.Series(mld_corr)
	smoothed_signal = signal_series.rolling(window=window_size, min_periods=1).mean().to_numpy()
	time_ds = inst.ds['time'][:].data
	time_dives = []
	for i in np.unique(inst.dives):
	    time_dives.append(np.mean(time_ds[inst.dives == i]))
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(time_dives, smoothed_signal, '-o', label='MLD Estimation')
	ax2.plot(aux.time, aux.era, label='ERA5 Wind Speed', color='orange')
	lines_1, labels_1 = ax1.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
	ax1.set_ylabel('MLD Depth (m)')
	ax2.set_ylabel('Wind Speed (m/s)')
	plt.grid()
	plt.show()
'''

"""depids_with_mld = []
for depid in depids[-1:] :
	df = pd.read_csv(os.path.join(path, depid, f'{depid}_dive.csv'))
	try :
		if np.all(np.isnan(df.meop_mld.to_numpy())):
			pass
		else :
			continue
	except AttributeError:
		pass
	inst = MixedLayerDepth(depid, path = os.path.join(path, depid))
	time_mld, mld = inst.ctd_mld(f'D:/individus_brut/MEOP_profiles/meop_{depid}.nc')
	insert = np.searchsorted(df.end_time, time_mld[(time_mld > df.begin_time.iloc[0]) & (time_mld < df.end_time.iloc[-1])])
	mld = np.full(len(df), np.nan)
	mld[insert] = mld[(time_mld > df.begin_time.iloc[0]) & (time_mld < df.end_time.iloc[-1])]
	df['meop_mld'] = mld
	df.to_csv(os.path.join(path, depid, f'{depid}_dive.csv'), index = None)"""