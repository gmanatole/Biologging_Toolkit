import netCDF4 as nc
import os
import numpy as np

class Wrapper():

	dt : int = 3

	def __init__(
		self, 
		ind : str = None,
		path : str = None,
	):
		"""
		Parameters
		----------
		ind : 'str'
			str describing a unique individual (eg. ml17_280a). Defaults to None.
		path : 'str'
			Path to directory where final dataset will be saved. Defaults to current working directory.
		dt : 'int'
			Timestep for entire dataset. Please choose N carefully, for instance computing Euler angles using inertial data requires N between 2 and 5. 
		"""

		self.ind = ind
		self.path = path if path else os.getcwd()
		assert isinstance(self.ind, str), "Please specify an individual"

		#Check if individual file already exists, if not create it
		mode='a' if os.path.exists(os.path.join(self.path, self.ind + '.nc')) else 'w'
		self.ds = nc.Dataset(os.path.join(self.path, self.ind + '.nc'), mode = mode)
		self.dataset_name

	@property
	def dataset_name(self) :
		"""
		Get or set the dataset title and subtitle.
		The title and subtitle of the NetCDF dataset are set based on the instance's `ind` and `dt` attributes.
		"""
		title = getattr(self.ds, 'title', None)
		if title is None:
			self.ds.title = f"Processed dataset for {self.ind}"
			self.ds.subtitle = f"NetCDF structure storing processed data from the {self.ind} individual using a {self.dt} s timestep"
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

	def create_time(self, time_data, overwrite = False):
		"""
		Create or update the 'time' variable in the dataset.
		This method creates a new 'time' dimension and variable, or updates an existing one, with the provided time data.

		Parameters
		----------
		time_data : array-like
			The time data to store in the 'time' variable. Should be an array-like structure of float values.
		overwrite : bool, optional
			If True, any existing 'time' variable will be removed before creating the new one. Defaults to False.
		"""
		if overwrite :
			if 'time' in self.ds.variables:
				self.remove_variable('time')

		if 'time' not in self.ds.variables:
			time_dim = self.ds.createDimension('time', len(time_data)) # unlimited axis (can be appended to)
			time = self.ds.createVariable('time', np.float64, ('time',))
			time.units = 'seconds since 1970-01-01 00:00:00 UTC'
			time.long_name = 'POSIX timestamp'
			time.calendar = 'standard'
			time[:] = time_data

	def create_gps(self, lat_data, lon_data, overwrite = False):
		"""
		Create or update the 'lat' and 'lon' variables in the dataset.
		This method creates new 'lat' and 'lon' dimensions and variables, or updates existing ones, with the provided latitude and longitude data.

		Parameters
		----------
		lat_data : array-like
			The latitude data to store in the 'lat' variable. Should be an array-like structure of float values.
		lon_data : array-like
			The longitude data to store in the 'lon' variable. Should be an array-like structure of float values.
		overwrite : bool, optional
			If True, any existing 'lat' and 'lon' variables will be removed before creating the new ones. Defaults to False.
		"""
		if overwrite :
			if 'lat' in self.ds.variables:
				self.remove_variable('lat')
			if 'lon' in self.ds.variables:
				self.remove_variable('lon')

		if ('lat' not in self.ds.variables) & ('lon' not in self.ds.variables):
			lat_dim = self.ds.createDimension('lat', len(lat_data))     # latitude axis
			lon_dim = self.ds.createDimension('lon', len(lon_data))    # longitude axis
			lat = self.ds.createVariable('lat', np.float32, ('lat',))
			lat.units = 'decimal degrees north'
			lat.long_name = 'latitude'
			lon = self.ds.createVariable('lon', np.float32, ('lon',))
			lon.units = 'decimal degrees east'
			lon.long_name = 'longitude'
			lat[:] = lat_data
			lon[:] = lon_data

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
			new_file_path = os.path.join(self.path, self.ind + '_new.nc')
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
		os.rename(new_file_path, os.path.join(self.path, self.ind + '.nc'))
		self.ds = nc.Dataset(os.path.join(self.path, self.ind + '.nc'), mode='a', format='NETCDF4')





