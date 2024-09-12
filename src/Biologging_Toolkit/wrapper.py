import netCDF4 as nc
import os
import numpy as np
from scipy.interpolate import interp1d
from Biologging_Toolkit.utils.format_utils import *

class Wrapper():

	dt : int = 3

	def __init__(
		self, 
		depid : str = None,
		path : str = None,
	):
		"""
		Parameters
		----------
		depid : 'str'
			str describing a unique individual (eg. ml17_280a). Defaults to None.
		path : 'str'
			Path to directory where final dataset will be saved. Defaults to current working directory.
		dt : 'int'
			Timestep for entire dataset. Please choose N carefully, for instance computing Euler angles using inertial data requires N between 2 and 5. 
		"""

		self.depid = depid
		self.ds_name = depid + '_acoustic' if self.__class__.__name__ == 'Acoustic' else depid + '_sens'
		self.path = path if path else os.getcwd()
		assert isinstance(self.depid, str), "Please specify an individual"

		#Check if individual file already exists, if not create it
		mode='r+' if os.path.exists(os.path.join(self.path, self.ds_name + '.nc')) else 'w'
		self.ds = nc.Dataset(os.path.join(self.path, self.ds_name + '.nc'), mode = mode)
		self.ds.sampling_rate = self.dt
		self.ds.sampling_rate_units = 'Seconds'
		self.dataset_name

	@property
	def dataset_name(self) :
		"""
		Get or set the dataset title and subtitle.
		The title and subtitle of the NetCDF dataset are set based on the instance's `depid` and `dt` attributes.
		"""
		title = getattr(self.ds, 'title', None)
		if title is None:
			self.ds.title = f"Processed dataset for {self.depid}"
			self.ds.subtitle = f"NetCDF structure storing processed data from the {self.depid} individual using a {self.dt} s timestep"
		return title

	@dataset_name.setter
	def dataset_name(self, title : str) :
		"""
		Set the dataset title.
		Parameters
		----------
		title : str
			The title to set for the dataset.
		"""
		self.ds.title = title

	def create_time(self, time_data = None, time_path = None, overwrite = False, resample = True):
		"""
		Create or update the 'time' variable in the dataset.
		This method creates a new 'time' dimension and variable, or updates an existing one, with the provided time data.

		Parameters
		----------
		time_data : array-like
			The time data to store in the 'time' variable. Should be an array-like structure of float values. Optional.
		time_path : str
			Provide path to sens5 structure to create time array.
		overwrite : bool, optional
			If True, any existing 'time' variable will be removed before creating the new one. Defaults to False.
		"""
		if overwrite :
			if 'time' in self.ds.variables:
				self.remove_variable('time')
	
		if time_data is not None :	
			if resample :
				time_data = np.arange(time_data[0], time_data[-1], self.dt)

		if time_path :
			ds = nc.Dataset(time_path)
			length = len(ds['P'][:])
			ds_sr = ds['P'].sampling_rate
			time_data = get_start_time_sens(ds.dephist_device_datetime_start) + np.arange(0, length/ds_sr, np.round(1/ds_sr,2))

		if 'time' not in self.ds.variables:
			time_dim = self.ds.createDimension('time', len(time_data)) # unlimited axis (can be appended to)
			time = self.ds.createVariable('time', np.float64, ('time',))
			time.units = 'seconds since 1970-01-01 00:00:00 UTC'
			time.long_name = 'POSIX timestamp'
			time.calendar = 'standard'
			time[:] = time_data

	def create_gps(self, lat_data, lon_data, time_data = None, overwrite = False):
		"""
		Create or update the 'lat' and 'lon' variables in the dataset.
		This method creates new 'lat' and 'lon' dimensions and variables, or updates existing ones, with the provided latitude and longitude data.

		Parameters
		----------
		lat_data : array-like
			The latitude data to store in the 'lat' variable. Should be an array-like structure of float values.
		lon_data : array-like
			The longitude data to store in the 'lon' variable. Should be an array-like structure of float values.
		time_data : array-like, optional
			Time data corresponding to the lat, lon data in POSIX timestamps. If None, defaults to the dataset's main time data
		overwrite : bool, optional
			If True, any existing 'lat' and 'lon' variables will be removed before creating the new ones. Defaults to False.
		"""
		if time_data is None:
			assert len(self.ds['time'][:].data) == len(lat_data), 'Please provide the timestamps for the position data'
			time_data = self.ds['time'][:].data
			
		if overwrite :
			if 'lat' in self.ds.variables:
				self.remove_variable('lat')
			if 'lon' in self.ds.variables:
				self.remove_variable('lon')
		
		if len(time_data) != len(self.ds['time']) :
			lat_data = interp1d(time_data, lat_data, bounds_error=False, fill_value=np.nan)(self.ds['time'][:].data)
			lon_data = interp1d(time_data, lon_data, bounds_error=False, fill_value=np.nan)(self.ds['time'][:].data)

		if ('lat' not in self.ds.variables) & ('lon' not in self.ds.variables):
			lat = self.ds.createVariable('lat', np.float32, ('time',))
			lat.units = 'decimal degrees north'
			lat.long_name = 'latitude'
			lon = self.ds.createVariable('lon', np.float32, ('time',))
			lon.units = 'decimal degrees east'
			lon.long_name = 'longitude'
			lat[:] = lat_data
			lon[:] = lon_data


	def create_variable(self, var_name, var_data, time_data, overwrite=False, **kwargs):
		"""
		Create or update a variable in the dataset.
		This method creates a new variable or updates an existing one with the provided data and metadata attributes.
		It also checks that the input time data matches the existing time data in the dataset.

		Parameters
		----------
		var_name : str
		The name of the variable to create or update.
		var_data : array-like
		The data to store in the variable. Should be an array-like structure of values.
		time_data : array-like
		The time data corresponding to the variable data. Should match the existing time data in the dataset.
		overwrite : bool, optional
		If True, any existing variable with the same name will be removed before creating the new one. Defaults to False.
		**kwargs : dict, optional
		Additional attributes for the variable. For example, `units='units'`, `long_name='Variable Name'`.
		"""


		# Overwrite the variable if it already exists
		if overwrite and var_name in self.ds.variables:
			self.remove_variable(var_name)

		# Create the dimension for time if not already present
		if 'time' not in self.ds.dimensions:
			time_data = np.arange(time_data[0], time_data[-1], self.dt)
			self.ds.createDimension('time', len(time_data))

		#Interpolation to get same timestep as reference dataframe
		interp = interp1d(time_data, var_data, bounds_error = False)
		var_data = interp(self.ds['time'][:].data)

		# Create or update the variable
		if var_name not in self.ds.variables:
			var = self.ds.createVariable(var_name, np.float32, ('time',))
			var_attrs = {
				'units': 'unknown',
				'long_name': var_name
				}
			var_attrs.update(kwargs)
			for attr_name, attr_value in var_attrs.items():
				setattr(var, attr_name, attr_value)
			var[:] = var_data
		else:
			raise ValueError(f"Variable '{var_name}' already exists in the dataset. Use `overwrite=True` to replace it.")

		# Assign time data to the 'time' variable if not already present
		if 'time' not in self.ds.variables:
			time = self.ds.createVariable('time', np.float64, ('time',))
			time.units = 'seconds since 1970-01-01 00:00:00 UTC'
			time.long_name = 'POSIX timestamp'
			time.calendar = 'standard'
			time[:] = time_data


	def remove_variable(self, var_name: str):
		"""
		Removes a specified variable from the NetCDF dataset.
		This method creates a new NetCDF file with the same dimensions and attributes as the 
		current dataset, but excludes the specified variable.
		Parameters
		----------
		var_name : str
			The name of the variable to remove from the dataset.
		"""
		if var_name in self.ds.variables:
			# Create a new dataset and copy the variables excluding the one to remove
			new_file_path = os.path.join(self.path, self.ds_name + '_new.nc')
			with nc.Dataset(new_file_path, 'w', format='NETCDF4') as new_ds:
				# Copy dimensions
				for name, dimension in self.ds.dimensions.items():
					if name != var_name:
						new_ds.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
				# Copy variables except the one to be removed
				for name, variable in self.ds.variables.items():
					if name != var_name:
						new_var = new_ds.createVariable(name, variable.datatype, variable.dimensions)
						new_var[:] = variable[:]
				# Copy global attributes
				for attr_name in self.ds.ncattrs():
					setattr(new_ds, attr_name, getattr(self.ds, attr_name))

		# Replace old file with new file
		self.ds.close()
		os.rename(new_file_path, os.path.join(self.path, self.ds_name + '.nc'))
		self.ds = nc.Dataset(os.path.join(self.path, self.ds_name + '.nc'), mode='a', format='NETCDF4')






