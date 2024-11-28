import numpy as np
from glob import glob
import os
import soundfile as sf
from Biologging_Toolkit.wrapper import Wrapper
from Biologging_Toolkit.utils.acoustic_utils import *
from Biologging_Toolkit.utils.format_utils import *

class Surface(Wrapper) :

    threshold = 10

    def __init__(self,
                 depid,
                 *,
                 path,
                 wav_path = None
                 ) :

        super().__init__(
                     depid,
                     path
                     )

        self.wav_path = wav_path

    def get_surface_wav(self) :
            dives = self.ds['dives'][:].data
            depth = self.ds['depth'][:].data
            depth_mask = depth < self.threshold
            ref_time = self.ds['time'][:].data

            matches = (self.ds['time'][:].data[:, None] >= self.wav_start_time) & (self.ds['time'][:].data[:, None] <= self.wav_end_time)
            indices = np.where(matches.any(axis=1), matches.argmax(axis=1), -1)
            time_diffs = np.where(indices != -1, self.ds['time'][:].data - self.wav_start_time[indices], np.nan)
            dive_files = self.wav_fns[indices]
            for dive in np.unique(dives)[:10] :
                _time_diffs = time_diffs[depth_mask & (dives == dive)]
                surface_time = ref_time[depth_mask & (dives == dive)]
                surface_wavs = dive_files[depth_mask & (dives == dive)]
                data, _ = sf.read(fn, 
                                    start = int(_time_diffs[0]*self.samplerate),
                                    stop = int((_time_diffs[-1]+self.params['duration'])*self.samplerate),
                                    dtype = 'float32')


    @staticmethod
    def get_surface_parameters(sig, fs) :
        pass


    def get_timestamps(self, timestamp_path = None, from_raw = False) :
        if from_raw :
            self.wav_fns = np.array(glob(os.path.join(self.wav_path, '*wav')))
            xml_fns = np.array(glob(os.path.join(self.wav_path, '*xml')))
            xml_fns = xml_fns[xml_fns != glob(os.path.join(self.wav_path, '*dat.xml'))]
            self.wav_start_time = get_start_date_xml(xml_fns)
            wav_end_time = []
            for file in self.wav_fns :
                wav_end_time.append(sf.info(file).duration)
            wav_end_time = np.array(wav_end_time) + self.wav_start_time
            self.wav_end_time = wav_end_time
        else :
            _timestamp = get_epoch(pd.read_csv(timestamp_path))
            self.wav_fns = np.array([os.path.join(self.wav_path, elem) for elem in _timestamp.filename.to_numpy()])
            self.wav_start_time = _timestamp.epoch.to_numpy(dtype = np.float64)
            self.wav_end_time = np.array([self.wav_start_time[i] + sf.info(self.wav_fns[i]).duration for i in range(len(self.wav_fns))])