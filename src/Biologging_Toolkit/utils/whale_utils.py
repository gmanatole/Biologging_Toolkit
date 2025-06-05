import pandas as pd
import os
import numpy as np
def successive_detections(series):
    changes = series.diff().fillna(0)
    starts = changes == 1
    ends = changes == -1
    cumulative = series.cumsum()
    lengths = cumulative[ends].diff().fillna(cumulative[ends])
    result = np.zeros_like(series)
    result[ends[ends].index] = lengths.values
    return pd.Series(result, index=series.index)

def sliding_window_sum(time_series, N):
    result = np.zeros_like(time_series)
    for i in range(len(time_series) - N + 1):
        result[i + N - 1] = np.sum(time_series[i:i + N])
    return result

def format_timestamps(depids, annotation_path):
    for depid in depids:
        try:
            annot = pd.read_csv(os.path.join(annotation_path, depid, 'timestamps.csv'), delimiter=',')
            if annot.shape[1] == 1:
                raise ValueError("Probably wrong delimiter")
        except:
            annot = pd.read_csv(os.path.join(annotation_path, depid, 'timestamps.csv'), delimiter=';')
        for col in ['Annotation', 'Annotation2', 'Annotation3']:
            _col = annot[col].astype(str).to_numpy()
            _col[np.isin(_col, ['Antarctic Blue Whale', 'Antarctic blue whale', 'Blue whale ', 'Blue whale', 'Dcall'])] = 'ABW'
            _col[np.isin(_col, ['Fin whale'])] = 'FW'
            _col[np.isin(_col, ['Humpback whale'])] = 'HW'
            _col[np.isin(_col, ['Minke whale'])] = 'MW'
            _col[np.isin(_col, ['Sei whale'])] = 'SW'
            _col[np.isin(_col, ['Southern Right Whale', 'Southern Right Whales'])] = 'SRW'
            _col[np.isin(_col, ['delphinid click', 'Delphinid click', 'Odontocete clicks'])] = 'Delphinid clicks'
            _col[np.isin(_col, ['Odontocete buzz', 'Buzz', 'Unknown buzz'])] = 'Buzz'
            _col[np.isin(_col,
                         ['Odontocete whistle', 'Unindentified whistle', 'delphinid whistle'])] = 'Delphinid whistle'
            _col[np.isin(_col, ['Unidentified clicks', 'Unindentified clicks'])] = 'Unidentified clicks'
            annot[col] = _col
        annot.to_csv(os.path.join(annotation_path, depid, 'formatted_timestamps.csv'), index=None)