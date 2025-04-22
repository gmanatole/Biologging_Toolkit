import copernicusmarine
from datetime import datetime
import pandas as pd

dataset = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
dataset = "cmems_mod_glo_phy_myint_0.083deg_P1D-m"
variables = ["so", "thetao"]
output_directory = ""
depth_min, depth_max = 0, 6000

lat_min, lat_max = -48, -40
lon_min, lon_max = 60, 70
date_start = datetime(2021,5,3,00,00,00).strftime('%Y-%m-%dT%H:%M:%S')
date_end =  datetime(2021,8,4,00,00,00).strftime('%Y-%m-%dT%H:%M:%S')
copernicusmarine.subset(
        dataset_id=dataset,
        variables=variables,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=date_start,
        end_datetime=date_end,
        minimum_depth=0,
        maximum_depth=depth_max,
        output_directory = output_directory)