from datetime import datetime, timedelta, timezone
import os
import netCDF4
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Exemple d'utilisation :
# -----------------------

# depids = ['ml18_294b','ml19_292a','ml19_292b','ml19_293a','ml19_294a','ml20_293a','ml20_296b','ml20_313a','ml21_295a','ml21_305b'] # pas fonctionné : 
# for depid in depids :
#     Api_GPM.compile_gpm_data(depid, "E:/individus_filtered/", "E:/Downloads/GPM_data/")
#     Api_GPM.join_gpm(depid, f"E:/individus_filtered", "precipitation")
#
# -----------------------

def round_to_half_hour(dt):
    minutes = (dt.minute // 30) * 30
    rounded_dt = dt.replace(minute=minutes, second=0, microsecond=0)
    return rounded_dt

def add_29m_59s(dt):
    new_dt = dt + timedelta(minutes=29, seconds=59)
    return new_dt

def get_gpm_date_format(datetime):
    return datetime.strftime("%Y%m%d-S%H%M%S") + "-E" + add_29m_59s(datetime).strftime("%H%M%S")+f".{datetime.hour * 60 + datetime.minute:04d}"

def get_gpm_file_name(start_time_eleph, end_time_eleph):
    """ 
    start_time_eleph : timestamp de la première acquisition elephant
    end_time_eleph : timestamp de la dernière acquisition elephant
    """

    start_time = round_to_half_hour(start_time_eleph)
    end_time = round_to_half_hour(end_time_eleph)

    delta = timedelta(minutes=30)
    dates = []

    current = start_time
    while current <= add_29m_59s(end_time):
        dates.append(get_gpm_date_format(current))
        current += delta

    return dates

def compile_gpm_data(depid, fp, gpm_files_folder_path):
    """
    attribute :
    -----------

    depid : "mlxx_xxxy"
    fp : path to the folder containing the depid folder
    gpm_files_folder_path


    Structure :
    -----------

    fp -- 
        |-- depid0
        |-- depid1
        |-- ...


    gpm_files_folder_path
     |-- 3B-HHR-L.MS.MRG.3IMERG.20230901-S000000-E002959.0000-0000.nc4
     |-- 3B-HHR-L.MS.MRG.3IMERG.20230901-S003000-E005959.0000-0000.nc4
     |-- ...
    """
    print(f"-> {depid}")
    path = fp+"/"+depid

    nc = netCDF4.Dataset(path+f"/{depid}_sens.nc")
    first_timestamp = datetime.fromtimestamp(int(nc.variables['time'][0]), tz=timezone.utc) # Premier timestamp
    last_timestamp = datetime.fromtimestamp(int(nc.variables['time'][-1]), tz=timezone.utc)  # Dernier timestamp

    gpm_file_list = get_gpm_file_name(first_timestamp, last_timestamp)
    # nc_gpm_formatted = netCDF4.Dataset(os.path.join(fp,depid,f'gpm_{depid}.nc'), mode='w')
    nc.close()

    gpm_global_lat = None
    gpm_global_lon = None
    gpm_global_time = []
    gpm_global_precipitation = []

    for gpm_file in tqdm(gpm_file_list):
        # Ouvrir chaque fichier GPM
        gpm_file_path = f"{gpm_files_folder_path}/3B-HHR-L.MS.MRG.3IMERG.{gpm_file}.V07B.HDF5.nc4"

        if os.path.exists(gpm_file_path):
            gpm_nc = netCDF4.Dataset(gpm_file_path, mode='r')
            
            # Extraire les coordonnées latitude et longitude (une seule fois, elles sont les mêmes dans chaque fichier)
            if gpm_global_lat is None or gpm_global_lon is None:
                gpm_global_lat = gpm_nc.variables['lat'][:]
                gpm_global_lon = gpm_nc.variables['lon'][:]
            
            # Extraire le temps et les précipitations
            gpm_time = gpm_nc.variables['time'][:]
            gpm_precipitation = gpm_nc.variables['precipitation'][:]
            
            gpm_precipitation = np.transpose(gpm_precipitation, (0, 2, 1))

            # Ajouter les données du fichier GPM à nos listes globales
            gpm_global_time.append(gpm_time)
            gpm_global_precipitation.append(gpm_precipitation)
            
            # Fermer le fichier GPM après traitement
            gpm_nc.close()
        else :
            print(f"File {gpm_file_path} does not exist.")

    # Convertir les listes en arrays numpy pour les manipulations ultérieures
    gpm_global_time = np.concatenate(gpm_global_time, axis=0)
    gpm_global_precipitation = np.concatenate(gpm_global_precipitation, axis=0)

    # À ce point, gpm_global_lat, gpm_global_lon, gpm_global_time, gpm_global_precipitation contiennent les données combinées
    # Maintenant, vous pouvez écrire ces données dans un fichier netCDF global
    with netCDF4.Dataset(os.path.join(path,f'{depid}_gpm.nc'), 'w', format='NETCDF4') as gpm_global_nc:
        # Définir les dimensions (latitude, longitude, time)
        gpm_global_nc.createDimension('lat', len(gpm_global_lat))
        gpm_global_nc.createDimension('lon', len(gpm_global_lon))
        gpm_global_nc.createDimension('time', len(gpm_global_time))
        
        # Créer les variables pour les latitudes, longitudes, temps et précipitations
        latitudes = gpm_global_nc.createVariable('lat', 'f4', ('lat',))
        longitudes = gpm_global_nc.createVariable('lon', 'f4', ('lon',))
        time = gpm_global_nc.createVariable('time', 'f4', ('time',))
        precipitation = gpm_global_nc.createVariable('precipitation', 'f4', ('time', 'lat', 'lon'))
        
        # Assigner les données aux variables
        latitudes[:] = gpm_global_lat
        longitudes[:] = gpm_global_lon
        time[:] = gpm_global_time
        precipitation[:] = gpm_global_precipitation

def plot_GPM_value(depid, path, value, ref = 'begin_time', saveAsGif = False, savePath= '.', **kwargs):
	""" 
	To make this method works you need to copy/paste the line bellow in your jupyter notebook import cell
	%matplotlib widget 
	
	"""
	
	default = {'units':'unknown', 'long_name':value}
	attrs = {**default, **kwargs}
	gpm_ds = netCDF4.Dataset(os.path.join(path, f'{depid}_gpm.nc'), mode='r')
	
	if os.path.exists(os.path.join(path, f'{depid}_sens.nc')):
		nc = netCDF4.Dataset(os.path.join(path, f'{depid}_sens.nc'), mode='r')
		lat_min = np.nanmin(nc.variables['lat'][:])
		lat_max = np.nanmax(nc.variables['lat'][:])
		lon_min = np.nanmin(nc.variables['lon'][:])
		lon_max = np.nanmax(nc.variables['lon'][:])
		nc.close()
	else: 
		lon_min = interp_gpm.grid[2].min(),  # longitude min
		lon_max = interp_gpm.grid[2].max(),  # longitude max
		lat_min = interp_gpm.grid[1].min(),  # latitude min
		lat_max = interp_gpm.grid[1].max()

	base_time = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)

	time_gpm = np.array([
		(base_time + timedelta(seconds=int(second))).timestamp() 
		for second in gpm_ds['time'][:].data
	])
	interp_gpm = RegularGridInterpolator((time_gpm, gpm_ds['lat'][:].data, gpm_ds['lon'][:].data), gpm_ds[value][:].data, bounds_error = False)
	
	# lon_grid, lat_grid = np.meshgrid(interp_gpm.grid[1], interp_gpm.grid[2])
	vmin, vmax = interp_gpm.values.min(), interp_gpm.values.max()

	def update(val):
		time_index = int(slider_time.val)
		ax.clear()
		sc = ax.imshow(
			interp_gpm.values[time_index, :, :],
			cmap='viridis',
			vmin=vmin,
			vmax=vmax,
			origin='lower',
			extent=[
				lon_min, lon_max, lat_min, lat_max
			]
		)

		# plt.colorbar(sc, ax=ax) # pas trouvé le moyen de l'afficher sans que ça ne soit buggé
		ax.set_title(f'GPM total precipitation around {depid}: {datetime.fromtimestamp(interp_gpm.grid[0][time_index], tz=timezone.utc).strftime("%d-%B") }')
		plt.draw()

	fig, ax = plt.subplots()
	plt.subplots_adjust(bottom=0.25)

	ax_time = plt.axes([0.1, 0.1, 0.8, 0.03])
	slider_time = Slider(ax_time, 'Time', 0, len(time_gpm) - 1, valinit=0)
	slider_time.on_changed(update)

	def animate(frame):
		slider_time.set_val(frame % len(time_gpm))

	ani = FuncAnimation(fig, animate, frames=len(time_gpm), interval=10, repeat=False)

	update(0)

	if saveAsGif:
		writer = PillowWriter(fps=20)
		ani.save(os.path.join(savePath,f"rain_gpm_{depid}_imshow.gif"), writer=writer)
	if not saveAsGif:
		plt.show()
            
def join_gpm(depid, path, value, ref = 'begin_time', **kwargs):
	default = {'units':'unknown', 'long_name':value}
	attrs = {**default, **kwargs}
	gpm_path = os.path.join(path, depid,f"{depid}_gpm.nc")
	gpm_ds = netCDF4.Dataset(gpm_path)
	base_time = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)

	time_gpm = np.array([
		(base_time + timedelta(seconds=int(second))).timestamp() 
		for second in gpm_ds['time'][:].data
	])
	interp_gpm = RegularGridInterpolator((time_gpm, gpm_ds['lat'][:].data, gpm_ds['lon'][:].data), gpm_ds[value][:].data, bounds_error = False)
	
	file_path = os.path.join(path, depid, f"{depid}_sens.nc")
	print(f"Trying to open: {file_path}")

	if not os.path.exists(file_path):
		print(f"File not found: {file_path}")
		return

	try:
		file_path = os.path.join(path, depid, f"{depid}_sens.nc")
		print(f"Trying to open: {file_path}")
		with netCDF4.Dataset(file_path, mode='a') as ds:
			if value + "_GPM" in ds.variables:
				print(f"Variable {value + '_GPM'} already exists. Skipping NetCDF write.")
			else:
				var_data = interp_gpm((ds['time'][:].data, ds['lat'][:].data, ds['lon'][:].data))
				var = ds.createVariable(value + "_GPM", np.float32, ('time',))
				var_attrs = attrs.copy()
				var_attrs.update(kwargs)
				for attr_name, attr_value in var_attrs.items():
					setattr(var, attr_name, attr_value)
				var[:] = var_data
	except Exception as e:
		print(f"Error handling NetCDF: {e}")
	
	try :
		dive_ds = pd.read_csv(os.path.join(path, depid, f'{depid}_dive.csv'))
		lat_interp = interp1d(ds['time'][:].data, ds['lat'][:].data)
		lon_interp = interp1d(ds['time'][:].data, ds['lon'][:].data)
		dive_ds['lat'] = lat_interp(dive_ds[ref])
		dive_ds['lon'] = lon_interp(dive_ds[ref])
		dive_ds[value+"_GPM"] = interp_gpm((dive_ds[ref], dive_ds.lat, dive_ds.lon))
		dive_ds.to_csv(os.path.join(path, depid, f'{depid}_dive.csv'), index = None)
	except (FileNotFoundError, KeyError):
		print('No dive dataframe found')
	ds.close()
	gpm_ds.close()
