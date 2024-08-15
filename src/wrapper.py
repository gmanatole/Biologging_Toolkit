import netCDF4 as nc
import os
import numpy as np

class Wrapper():

	def __init__(
		self, 
		ind : str = None,
		path : str = None,
		dt : int = 3,
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
		self.dt = dt
		assert isinstance(self.ind, str), "Please specify an individual"

		#Check if individual file already exists, if not create it
		self.ds = nc.Dataset(os.path.join(self.path, self.ind + '.nc'), mode='a' if os.path.exists(os.path.join(self.path, self.ind + '.nc')) else 'w', format='NETCDF4_CLASSIC')


	@property
	def dataset_name(self) :
		self.ds.title = f"Processed dataset for {self.ind}"
		self.ds.subtitle = f"NetCDF structure storing processed data from the {self.ind} individual using a {self.dt} s timestep"

	@dataset_name.setter
	def dataset_name(self, title : str) :
		self.ds.title = title

	def create_time(self, time_data):
		time_dim = self.ds.createDimension('time', len(time_data)) # unlimited axis (can be appended to)
		time = self.ds.createVariable('time', np.float64, ('time',))
		time.units = 'seconds since 1970-01-01 00:00:00 UTC'
		time.long_name = 'POSIX timestamp'
		time.calendar = 'standard'
		time[:] = time_data

		date_dim = self.ds.createDimension('date', len(time_data))
		date = self.ds.createVariable('date', np.float64, ('date',))
		date.units = 'ISO formatted datetime in UTC'
		date.long_name = 'Datetime'
		date.calendar = 'standard'
		date[:] = [nc.num2date(time_value, units='seconds since 1970-01-01 00:00:00 UTC').isoformat() for time_value in time_data]
		
	def create_gps(self, lat_data, lon_data):
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





