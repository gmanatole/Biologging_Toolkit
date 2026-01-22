import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt
from Biologging_Toolkit.utils.inertial_utils import *
from Biologging_Toolkit.utils.format_utils import *
from Biologging_Toolkit.applications.Drift_Dives import DriftDives
from Biologging_Toolkit.processing.Dives import Dives

class Density(DriftDives) :

    """
    Class to estimate SES density. Methods are based on acceleration and vertical speed.
    Drift method : Computes drift phase vertical speed
    Descent method : Computed descent maximum speed
    Ascent method : Computes ascent swim effort
    """

    Cd = 0.69
    p_sw = 1027.5

    def __init__(
            self,
            depid,
            *,
            path,
            sens_path: str = None,
    ):
        '''
        Initializes an instance of the Density class.

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
            path = path
        )

        self.sens_path = sens_path


    def get_drift_speed(self, method = 'inertial', sens_path = None) :
        """
        This method computed the vertical speed during drift phases
        Both inertial (bank angle) or acceleration and depth can be used to detect drift phases
        The vertical speed during these detected drifts is then computed
        """
        if method == 'inertial' and 'inertial_drift' not in self.ds.variables.keys() :
            self.forward(mode = 'inertial')
        elif method == 'acceleration' and ('acc_drift' not in self.ds.variables.keys() and 'depth_drift' not in self.ds.variables.keys()) :
            self.sens_path = sens_path
            self.forward(mode = 'acceleration')
            self.forward(mode = 'depth')
        if method == 'inertial' :
            drifts = self.ds['inertial_drift'][:].data
        elif method == 'acceleration' :
            drifts = self.ds['acc_drift'][:].data.astype(bool) & self.ds['depth_drift'][:].data.astype(bool)
            drifts = drifts.astype(int)
        depth = self.ds['depth'][:].data
        dives = self.ds['dives'][:].data
        _time = self.ds['time'][:].data
        Udrift, time_drift = [], []
        for dive in np.unique(dives):
            drift = drifts[dives == dive]
            profile = depth[dives == dive]
            profile_time = _time[dives == dive]
            phase = profile[drift == 1]
            phase_time = profile_time[drift == 1]
            Udrift.append(np.mean(np.diff(phase) / np.diff(phase_time)))
            time_drift.append(np.mean(phase_time))
        self.Udrift = - np.array(Udrift)
        self.time_Udrift = np.array(time_drift)

    def get_descent_speed(self) :
        if 'descent' not in self.__dir__() :
            ascent, descent = Dives.get_dive_direction(self.ds['depth'][:].data[::20])
            _time = self.ds['time'][:].data
            f = interp1d(_time[::60//int(1/self.ds.sampling_rate)], descent, bounds_error=False)
            descent_full = f(_time)
            self.descent = np.round(descent_full).astype(bool)
            f = interp1d(_time[::60//int(1/self.ds.sampling_rate)], ascent, bounds_error=False)
            ascent_full = f(_time)
            self.ascent = np.round(ascent_full).astype(bool)
        Udesc = []
        time_Udesc = []
        dives = self.ds['dives'][:].data
        depth = self.ds['depth'][:].data
        for dive in np.unique(dives):
            profile = depth[dives == dive]
            time_Udesc.append(np.mean(_time[dives == dive]))
            mask_dive = self.descent[dives == dive]
            descent_profile = profile[mask_dive]
            try:
                Udesc.append(np.max(np.diff(descent_profile))*self.ds.sampling_rate)
            except:
                Udesc.append(np.nan)
        self.time_Udesc = np.array(time_Udesc)
        self.Udesc = np.array(Udesc)

    def get_ascent_effort(self) :
        if 'ascent' not in self.__dir__() :
            ascent, descent = Dives.get_dive_direction(self.ds['depth'][:].data[::20])
            _time = self.ds['time'][:].data
            f = interp1d(_time[::60//int(1/self.ds.sampling_rate)], descent, bounds_error=False)
            descent_full = f(_time)
            self.descent = np.round(descent_full).astype(bool)
            f = interp1d(_time[::60//int(1/self.ds.sampling_rate)], ascent, bounds_error=False)
            ascent_full = f(_time)
            self.ascent = np.round(ascent_full).astype(bool)
        time_Uasc = []
        Afreq = []
        Ae = self.ds['Ax'][:].data
        sos = butter(N = 2, Ws = 0.3, btype = 'highpass', fs=self.ds.sampling_rate, output='sos')
        dives = self.ds['dives'][:].data
        depth = self.ds['depth'][:].data
        for dive in np.unique(dives):
            profile = depth[dives == dive]
            time_Uasc.append(np.mean(_time[dives == dive]))
            mask_dive = self.ascent[dives == dive]
            ascent_profile = profile[mask_dive]
            try:
                _Ae = Ae[ascent_profile]
                Ae_filt = sosfilt(sos, Ae)
                fft = np.fft.fft(Ae_filt)
                freq = np.fft.ffrfreq(Ae_filt)
                Afreq.append(np.nanmax(freq[np.argmax(fft)]))
            except:
                Afreq.append(np.nan)
        self.time_Uasc = np.array(time_Uasc)
        self.Afreq = np.array(Afreq)



    def water_density(self) :
        self.p_sw = np.nan

    def drift_density(self, S_seal, V_seal) :
        self.p_seal = self.p_sw + (self.Ud**2 * self.Cd * self.p_sw * S_seal) / (2 * 9.81 * V_seal)