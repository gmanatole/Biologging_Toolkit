import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import spectrogram
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
        This method computes the vertical speed during drift phases
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
        elevation_angle = self.ds['elevation_angle'][:].data
        _time = self.ds['time'][:].data
        Udrift, time_drift, ea_drift = [], [], []
        for dive in np.unique(dives):
            drift = drifts[dives == dive]
            profile = depth[dives == dive]
            profile_time = _time[dives == dive]
            phase = profile[drift == 1]
            phase_time = profile_time[drift == 1]
            Udrift.append(np.mean(np.diff(phase) / np.diff(phase_time)))
            time_drift.append(np.mean(phase_time))
            ea_drift.append(np.nanmean(elevation_angle[drift][drift == 1]))
        self.Udrift = - np.array(Udrift)
        self.time_Udrift = np.array(time_drift)
        self.ea_drift = np.array(ea_drift)

    def get_descent_speed(self) :
        """
        This method computes the maximal vertical speed during active swimming descent phases.
        The vertical speed during these detected drifts is computed using pressure data.
        """
        # Get descent portion of dives from Dives module
        if 'descent' not in self.__dir__() :
            ascent, descent = Dives.get_dive_direction(self.ds['depth'][:].data[::int(60/(1/self.ds.sampling_rate))])
            _time = self.ds['time'][:].data
            f = interp1d(_time[::int(60/(1/self.ds.sampling_rate))], descent, bounds_error=False)
            descent_full = f(_time)
            self.descent = np.round(descent_full).astype(bool)
            f = interp1d(_time[::int(60/(1/self.ds.sampling_rate))], ascent, bounds_error=False)
            ascent_full = f(_time)
            self.ascent = np.round(ascent_full).astype(bool)
        else :
            _time = self.ds['time'][:].data
        Udesc, time_Udesc = [], []
        dives = self.ds['dives'][:].filled(np.nan)
        depth = self.ds['depth'][:].filled(np.nan)
        #Iterate through dives removing dives containing NaN or with two descents detected
        for dive in np.unique(dives):
            profile = depth[dives == dive]
            time_Udesc.append(np.mean(_time[dives == dive]))
            mask_dive = self.descent[dives == dive]
            mask_dive[np.isnan(profile)] = False
            if (np.diff(mask_dive.astype(int)) == -1).sum() > 1 :
                Udesc.append(np.nan)
                continue
            descent_profile = profile[mask_dive]
            descent_profile = descent_profile[descent_profile < 350]  # Only focus on upper 350m
            try:
                Udesc.append(np.max(np.diff(descent_profile)) * self.ds.sampling_rate)
            except:
                Udesc.append(np.nan)
        self.time_Udesc = np.array(time_Udesc, dtype=float)
        self.Udesc = np.array(Udesc, dtype=float)

    def get_ascent_effort(self) :
        """
        This method computes the lateral oscillation period of the individual during ascent phases
        Lateral acceleration during ascent is isolated and the frequency components above 0.2 Hz are extracted.
        The median of the maxima of each time bin is the final oscillation period.
        """
        if 'ascent' not in self.__dir__() :
            ascent, descent = Dives.get_dive_direction(self.ds['depth'][:].data[::int(60/(1/self.ds.sampling_rate))])
            _time = self.ds['time'][:].data
            f = interp1d(_time[::int(60/(1/self.ds.sampling_rate))], descent, bounds_error=False)
            descent_full = f(_time)
            self.descent = np.round(descent_full).astype(bool)
            f = interp1d(_time[::int(60/(1/self.ds.sampling_rate))], ascent, bounds_error=False)
            ascent_full = f(_time)
            self.ascent = np.round(ascent_full).astype(bool)
        else :
            _time = self.ds['time'][:].data
        time_Uasc = []
        Afreq = []
        Ae = self.ds['Ay'][:]
        # High pass filter to remove low frequency movements
        sos = butter(N = 2, Wn = 0.3, btype = 'highpass', fs=self.ds.sampling_rate, output='sos')
        dives = self.ds['dives'][:].data
        depth = self.ds['depth'][:].data
        for dive in np.unique(dives):
            #Ensure elephant dives deep enough to have ascent data
            if np.nanmax(depth[dives == dive]) < 100 :
                continue
            _Ae = Ae[dives == dive]
            time_Uasc.append(np.mean(_time[dives == dive]))
            mask_dive = self.ascent[dives == dive]
            ascent_Ae = _Ae[mask_dive]
            # sosfilt is not compatible with nan values, extraction of longest data segment
            if np.isnan(ascent_Ae).sum() != 0:
                valid = ~np.isnan(ascent_Ae)
                edges = np.diff(np.concatenate(([0], valid.view(np.int8), [0])))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]
                lengths = ends - starts
                idx = np.argmax(lengths)
                if ends[idx] - starts[idx] < 0.5 * len(ascent_Ae):
                    continue
                else:
                    ascent_Ae = ascent_Ae[starts[idx]: ends[idx]]
            Ae_filt = sosfilt(sos, ascent_Ae)
            freq, t, Sxx = spectrogram(sosfilt(sos, Ae_filt), fs=5, nperseg=128, noverlap=60)
            Afreq.append(np.nanmedian(freq[np.argmax(Sxx, axis = 0)]))
        self.time_Uasc = np.array(time_Uasc)
        self.Afreq = np.array(Afreq)


    def water_density(self) :
        self.p_sw = np.nan

    def drift_density(self, S_seal, V_seal) :
        self.p_seal = self.p_sw + (self.Ud**2 * self.Cd * self.p_sw * S_seal) / (2 * 9.81 * V_seal)