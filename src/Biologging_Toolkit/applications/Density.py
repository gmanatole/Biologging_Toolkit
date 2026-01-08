import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm
from Biologging_Toolkit.utils.inertial_utils import *
from Biologging_Toolkit.utils.format_utils import *
from Biologging_Toolkit.wrapper import Wrapper


class Density(Wrapper) :

    """
    Class to estimate SES density. Methods are based on acceleration and vertical speed.
    """
    Cd = 0.69

    def __init__(
            self,
            depid,
            *,
            path,
            sens_path: str = None,
    ):
        '''
        Initializes an instance of the Inertial class, which handles loading and processing inertial sensor data.

        Parameters
        ----------
        depid : str
            Identifier corresponding to the individual for whom data is being loaded (e.g., 'ml17_280a').
        path : str
            The path to the main dataset file required by the superclass `Wrapper`.
        sens_data : str, optional
            Path to the inertial dataset file (e.g., containing magnetometer and accelerometer data).
            If provided, data will be loaded from this file. Default is None.
        '''

        super().__init__(
            depid,
            path
        )

        self.sens_path = sens_path


    def get_drift_speed(self) :
        self.Ud = np.nan

    def water_density(self) :
        self.p_sw = np.nan

    def drift_density(self, S_seal, V_seal) :
        self.p_seal = self.p_sw + (self.Ud**2 * self.Cd * self.p_sw * S_seal) / (2 * 9.81 * V_seal)