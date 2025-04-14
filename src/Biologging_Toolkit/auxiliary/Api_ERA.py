import cdsapi
from datetime import date, datetime
import os
import calendar
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter

#get_epoch_time = lambda x : calendar.timegm(x.timetuple()) if isinstance(x, datetime) else x

def make_cds_file(personnal_access_token, path):
	os.chdir(os.path.expanduser("~"))
	try :
 	   os.remove('.cdsapirc')
	except FileNotFoundError :
	    pass
	'''
	cmd1 = "echo url: https://cds.climate.copernicus.eu/api/v2 >> .cdsapirc"
	cmd2 = "echo key: {}:{} >> .cdsapirc".format(udi, key)
	os.system(cmd1)
	os.system(cmd2)
	'''
	# cmd1 = "echo url: https://cds-beta.climate.copernicus.eu/api >> .cdsapirc"
	cmd1 = "echo url: https://cds.climate.copernicus.eu/api >> .cdsapirc"
	cmd2 = "echo key: {} >> .cdsapirc".format(personnal_access_token)
	os.system(cmd1)
	os.system(cmd2)

	if path == None:
		try :
		   os.mkdir('api')
		except FileExistsError:
		    pass
		path_to_api = os.path.join(os.path.expanduser("~"), "api/")
	else :
		path_to_api = path

	os.chdir(path_to_api)
	os.getcwd()

def return_cdsbeta(filename, variables, years, months, days, hours, area):

	print('You have selected : \n')
	for data in variables :
		print(f'   - {data}')
	print('\nfor the following times')
	print(f'Years : {years} \n Months : {months} \n Days : {days} \n Hours : {hours}')

	print('\nYour boundaries are : North {}°, South {}°, East {}°, West {}°'.format(area[0], area[2],
			                                                         area[3], area[1]))
	filename = filename + '.zip'

	# c = cdsapi.Client(verify=False)
	c = cdsapi.Client(verify=False)

	if days == 'all':
		days = ['01', '02', '03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
	if months == 'all':
		months = ['01','02','03','04','05','06','07','08','09','10','11','12']
	if hours == 'all':
		hours = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']


	dataset = 'reanalysis-era5-single-levels'
	request = {
		'product_type': ['reanalysis'],
		'variable': variables,
		'year': years,
		'month': months,
		'day': days,
		'time': hours,
		'area' : area,
		'download_format': 'zip',
		'data_format': 'netcdf',
		}

	c.retrieve(dataset, request, filename)

def return_cdsv2(filename, key, variables, years, months, days, hours, area):

	print('You have selected : \n')
	sel = [print(variables) for data in variables]
	print('\nfor the following times')
	print(f'Years : {years} \n Months : {months} \n Days : {days} \n Hours : {hours}')

	print('\nYour boundaries are : North {}°, South {}°, East {}°, West {}°'.format(area[0], area[2],
			                                                         area[3], area[1]))

	filename = filename + '.nc'
	c = cdsapi.Client()

	if days == 'all':
		days = ['01', '02', '03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
	if months == 'all':
		months = ['01','02','03','04','05','06','07','08','09','10','11','12']
	if hours == 'all':
		hours = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']

	r = c.retrieve('reanalysis-era5-single-levels',
	    {
	      'product_type' : 'reanalysis',
	      'variable' : variables,
	      'year' : years,
	      'month' : months,
	      'day' : days,
	      'time' : hours,
	      'area' : area,
	      'format' : 'netcdf',
	      'grid':[0.25, 0.25],
	    },
	    filename,
	    )
	r.download(filename)


def join_era(depid, path, era_path, value, ref = 'begin_time', **kwargs):
	default = {'units':'unknown', 'long_name':value}
	attrs = {**default, **kwargs}
	era_ds = Dataset(era_path)
	time_era = np.array([(datetime(1900,1,1,0,0,0) + timedelta(hours = int(hour))).replace(tzinfo=timezone.utc).timestamp() for hour in era_ds['time'][:].data])
	interp_era = RegularGridInterpolator((time_era, era_ds['latitude'][:].data, era_ds['longitude'][:].data), era_ds[value][:].data, bounds_error = False)
	
	try :
		ds = Dataset(os.path.join(path, depid + '_sens.nc'), mode = 'a')
		var_data = interp_era((ds['time'][:].data, ds['lat'][:].data, ds['lon'][:].data))
		var = ds.createVariable(value, np.float32, ('time',))
		var_attrs = attrs
		var_attrs.update(kwargs)
		for attr_name, attr_value in var_attrs.items():
			setattr(var, attr_name, attr_value)
		var[:] = var_data
	except (FileNotFoundError, RuntimeError):
		print('No reference NetCDF file found')
	
	try :
		dive_ds = pd.read_csv(os.path.join(path, depid + '_dive.csv'))
		lat_interp = interp1d(ds['time'][:].data, ds['lat'][:].data)
		lon_interp = interp1d(ds['time'][:].data, ds['lon'][:].data)
		dive_ds['lat'] = lat_interp(dive_ds[ref])
		dive_ds['lon'] = lon_interp(dive_ds[ref])
		dive_ds[value] = interp_era((dive_ds[ref], dive_ds.lat, dive_ds.lon))
		dive_ds.to_csv(os.path.join(path, depid + '_dive.csv'), index = None)
	except (FileNotFoundError, KeyError):
		print('No dive dataframe found')
	ds.close()

def plot_era_value(depid, path, era_path, value, ref = 'begin_time', saveAsGif = False, savePath= '.',isGPMData=False, **kwargs):
	""" 
	To make this method works you need to copy/paste the line bellow in your jupyter notebook import cell
	%matplotlib widget 
	
	"""
	
	default = {'units':'unknown', 'long_name':value}
	attrs = {**default, **kwargs}
	era_ds = Dataset(era_path)
	
	if(isGPMData):
		base_time = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
    
		time_era = np.array([
			(base_time + timedelta(seconds=int(second))).timestamp() 
			for second in era_ds['time'][:].data
		])
	else :
		time_era = np.array([(datetime(1900,1,1,0,0,0) + timedelta(hours = int(hour))).replace(tzinfo=timezone.utc).timestamp() for hour in era_ds['time'][:].data])
	interp_era = RegularGridInterpolator((time_era, era_ds['latitude'][:].data, era_ds['longitude'][:].data), era_ds[value][:].data, bounds_error = False)
	
	lon_grid, lat_grid = np.meshgrid(interp_era.grid[1], interp_era.grid[2])
	vmin, vmax = interp_era.values.min(), interp_era.values.max()

	def update(val):
		time_index = int(slider_time.val)
		ax.clear()
		# sc = ax.scatter(lon_grid, lat_grid, c=interp_era.values[time_index, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
		sc = ax.imshow(
			interp_era.values[time_index, :, :],
			cmap='viridis',
			vmin=vmin,
			vmax=vmax,
			origin='lower',
			extent=[
				interp_era.grid[1].min(),  # longitude min
				interp_era.grid[1].max(),  # longitude max
				interp_era.grid[2].min(),  # latitude min
				interp_era.grid[2].max()   # latitude max
			]
		)

		# plt.colorbar(sc, ax=ax) # pas trouvé le moyen de l'afficher sans que ça ne soit buggé
		ax.set_title(f'Era5 total precipitation around {depid}: {datetime.fromtimestamp(interp_era.grid[0][time_index], tz=timezone.utc).strftime("%d-%B") }')
		plt.draw()

	fig, ax = plt.subplots()
	plt.subplots_adjust(bottom=0.25)

	ax_time = plt.axes([0.1, 0.1, 0.8, 0.03])
	slider_time = Slider(ax_time, 'Time', 0, len(time_era) - 1, valinit=0)
	slider_time.on_changed(update)

	def animate(frame):
		slider_time.set_val(frame % len(time_era))

	ani = FuncAnimation(fig, animate, frames=len(time_era), interval=10, repeat=False)

	update(0)

	if saveAsGif:
		writer = PillowWriter(fps=20)
		ani.save(os.path.join(savePath,f"rain_era5_{depid}_imshow.gif"), writer=writer)
	if not saveAsGif:
		plt.show()

def get_max_neighborhood(era_data, time_era, lats, lons, t, lat, lon, window=1):
	t_idx = np.abs(time_era - t).argmin()
	lat_idx = np.abs(lats - lat).argmin()
	lon_idx = np.abs(lons - lon).argmin()

	t_min = max(t_idx - window, 0)
	t_max = min(t_idx + window + 1, len(time_era))
	lat_min = max(lat_idx - window, 0)
	lat_max = min(lat_idx + window + 1, len(lats))
	lon_min = max(lon_idx - window, 0)
	lon_max = min(lon_idx + window + 1, len(lons))

	neighborhood = era_data[t_min:t_max, lat_min:lat_max, lon_min:lon_max]
	return np.nanmax(neighborhood)

def join_era_maxPool(depid, path, era_path, value, ref='begin_time', window=1, **kwargs):
	default = {'units': 'unknown', 'long_name': value}
	attrs = {**default, **kwargs}
	era_ds = Dataset(era_path)

	time_era = np.array([
		(datetime(1900, 1, 1) + timedelta(hours=int(hour)))
		.replace(tzinfo=timezone.utc).timestamp()
		for hour in era_ds['time'][:].data
	])
	lats = era_ds['latitude'][:].data
	lons = era_ds['longitude'][:].data
	era_data = era_ds[value][:].data

	try:
		ds = Dataset(os.path.join(path, depid + '_sens.nc'), mode='a')
		times = ds['time'][:].data
		lats_interp = ds['lat'][:].data
		lons_interp = ds['lon'][:].data

		print(f'Interpolating {value} on NetCDF data...')
		var_data = np.array([
			get_max_neighborhood(era_data, time_era, lats, lons, t, lat, lon, window)
			for t, lat, lon in tqdm(zip(times, lats_interp, lons_interp), total=len(times))
		])

		var = ds.createVariable(value+"maxPool", np.float32, ('time',))
		for attr_name, attr_value in attrs.items():
			setattr(var, attr_name, attr_value)
		var[:] = var_data
	except (FileNotFoundError, RuntimeError):
		print('No reference NetCDF file found')

	try:
		dive_ds = pd.read_csv(os.path.join(path, depid + '_dive.csv'))
		lat_interp = interp1d(times, lats_interp, bounds_error=False, fill_value="extrapolate")
		lon_interp = interp1d(times, lons_interp, bounds_error=False, fill_value="extrapolate")

		dive_ds['lat'] = lat_interp(dive_ds[ref])
		dive_ds['lon'] = lon_interp(dive_ds[ref])

		print(f'Interpolating {value} on dive.csv...')
		dive_ds[value+"maxPool"] = [
			get_max_neighborhood(era_data, time_era, lats, lons, t, lat, lon, window)
			for t, lat, lon in tqdm(zip(dive_ds[ref], dive_ds['lat'], dive_ds['lon']), total=len(dive_ds))
		]
		dive_ds.to_csv(os.path.join(path, depid + '_dive.csv'), index=None)
	except (FileNotFoundError, KeyError):
		print('No dive dataframe found')

	ds.close()


