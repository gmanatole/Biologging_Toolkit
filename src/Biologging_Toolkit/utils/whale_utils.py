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

def load_annotation_data(ker, arg, annotation_path) :
    data_arg = pd.DataFrame()
    for depid in arg:
        data_arg = pd.concat((data_arg, pd.read_csv(os.path.join(annotation_path, depid, 'formatted_timestamps.csv'))))
    data_arg['drift_time'] = data_arg.end_drift - data_arg.start_drift
    data_arg = data_arg[data_arg.drift_time > 120]
    arg_drift = data_arg.drift_time.to_numpy()
    idx_arg = data_arg[['Indice', 'Indice2', 'Indice3']].to_numpy().flatten()
    data_arg = data_arg[['Annotation', 'Annotation2', 'Annotation3']].astype(str).to_numpy().flatten()

    data_ker = pd.DataFrame()
    for depid in ker:
        data_ker = pd.concat((data_ker, pd.read_csv(os.path.join(annotation_path, depid, 'formatted_timestamps.csv'))))
    data_ker['drift_time'] = data_ker.end_drift - data_ker.start_drift
    data_ker = data_ker[data_ker.drift_time > 120]
    ker_drift = data_ker.drift_time.to_numpy()
    idx_ker = data_ker[['Indice', 'Indice2', 'Indice3']].to_numpy().flatten()
    data_ker = data_ker[['Annotation', 'Annotation2', 'Annotation3']].astype(str).to_numpy().flatten()
    return ker_drift, idx_ker, data_ker, arg_drift, idx_arg, data_arg
def get_number_of_annotation(ker, arg, annotation_path, idx = 0) :
    ker_drift, idx_ker, data_ker, arg_drift, idx_arg, data_arg = load_annotation_data(ker, arg, annotation_path)
    data_arg = data_arg[idx_arg >= idx]
    data_ker = data_ker[idx_ker >= idx]
    _label_arg, _count_arg = np.unique(data_arg, return_counts=True)
    _label_ker, _count_ker = np.unique(data_ker, return_counts=True)
    print('Kerguelen')
    print(pd.DataFrame({"Annotation":_label_ker, "Count": _count_ker}))
    print('Argentina')
    print(pd.DataFrame({"Annotation":_label_arg, "Count": _count_arg}))

def get_drift_time_positive_proportion(ker, arg, annotation_path, idx = 0) :

    ker_drift, idx_ker, data_ker, arg_drift, idx_arg, data_arg = load_annotation_data(ker, arg, annotation_path)
    total_ker = ker_drift.sum()

    ker_bal = ['ABW', 'FW', 'MW', 'SW', 'HW', 'Sweep', 'Downsweep', 'SRW']
    ker_odo = ['Buzz', 'Clicks', 'Delphinid clicks', 'Delphinid whistle']
    mask_ker = (idx_ker >= idx).reshape(-1, 3)
    _data_ker = data_ker.reshape(-1, 3).copy()
    _data_ker[~mask_ker] = np.nan
    print('KER WHALE : ',
          ker_drift[np.max(np.isin(_data_ker, ker_bal).astype(int), axis=1).astype(bool)].sum() / total_ker)
    print('KER ODONTOCETES : ',
          ker_drift[np.max(np.isin(_data_ker, ker_odo).astype(int), axis=1).astype(bool)].sum() / total_ker)
    print('KER SPERMWHALE : ',
          ker_drift[np.max(np.isin(_data_ker, ['Spermwhale']).astype(int), axis=1).astype(bool)].sum() / total_ker)

    total_arg = arg_drift.sum()
    arg_bal = ['ABW', 'FW', 'MW', 'SW', 'HW', 'Sweep', 'Downsweep', 'SRW']
    arg_odo = ['Buzz', 'Clicks', 'Delphinid clicks', 'Delphinid whistle']
    mask_arg = (idx_arg >= idx).reshape(-1, 3)
    _data_arg = data_arg.reshape(-1, 3).copy()
    _data_arg[~mask_arg] = np.nan
    print('ARG WHALE : ',
          arg_drift[np.max(np.isin(_data_arg, arg_bal).astype(int), axis=1).astype(bool)].sum() / total_arg)
    print('ARG ODONTOCETES : ',
          arg_drift[np.max(np.isin(_data_arg, arg_odo).astype(int), axis=1).astype(bool)].sum() / total_arg)
    print('ARG SPERMWHALE : ',
          arg_drift[np.max(np.isin(_data_arg, ['Spermwhale']).astype(int), axis=1).astype(bool)].sum() / total_arg)


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