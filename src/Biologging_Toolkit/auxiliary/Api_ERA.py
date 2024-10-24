import cdsapi
from datetime import date, datetime
import os
import calendar
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator

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
	cmd1 = "echo url: https://cds-beta.climate.copernicus.eu/api >> .cdsapirc"
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

	c = cdsapi.Client()

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



def join_era(depid, path, era_path, value, **kwargs):
	default = {'units':'unknown', 'long_name':value}
	attrs = {**default, **kwargs}
	era_ds = Dataset(era_path)
	time_era = np.array([(datetime(1900,1,1,0,0,0) + timedelta(hours = int(hour))).timestamp() for hour in era_ds['time'][:].data])
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
		ds.close()
	except (FileNotFoundError, RuntimeError):
		print('No reference NetCDF file found')
	
	try :
		dive_ds = pd.read_csv(os.path.join(path, depid + '_dive.csv'))
		dive_ds[value] = interp_era((dive_ds.time, dive_ds.latitude, dive_ds.longitude))
		dive_ds.to_csv(os.path.join(path, depid + '_dive.csv'), index = None)
	except (FileNotFoundError, KeyError):
		print('No dive dataframe found')
