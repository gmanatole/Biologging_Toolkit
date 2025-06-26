import os
import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import curve_fit
import sklearn.metrics as metrics
from tqdm import tqdm
from sympy import Symbol
from Biologging_Toolkit.config.config_weather import empirical_rain

def rain_scale(x):
    return next((i for i, limit in enumerate([0.3, 1.0, 4.0, 16.0, 50.0]) if x < limit), 5)

class DataLoader:
    def __init__(self, depid: Union[str, List], path: Union[str, List], acoustic_path: Union[str, List], df_data="csv"):
        if isinstance(depid, list):
            assert len(depid) == len(path) == len(acoustic_path), "Depids, Paths and acoustic paths must match depid length"
        else:
            depid = [depid]
            path = [path]
            acoustic_path = [acoustic_path]

        self.depid = depid
        self.path = path
        self.acoustic_path = acoustic_path
        self.df_data = df_data

    def load(self):
        if self.df_data == "csv":
            fns, _dep = [], []
            df = pd.DataFrame({})
            for p, dep, ac_path in zip(self.path, self.depid, self.acoustic_path):
                df_csv = pd.read_csv(f"{p}/{dep}_dive.csv")
                df = pd.concat([df, df_csv], ignore_index=True)
                for i, row in df_csv.iterrows():
                    _dep.append(dep)
                    npz_path = os.path.join(ac_path, f'{dep}_dive_{int(row.dive):05d}.npz')
                    fns.append(npz_path if os.path.exists(npz_path) else 'N/A')
            df['fns'] = fns
            df['depid'] = _dep
            return df

        elif self.df_data == "npz_reduced":
            rows = []
            df = pd.DataFrame({})
            for p, dep in zip(self.path, self.depid):
                df_csv = pd.read_csv(f"{p}/{dep}_dive.csv")
                for idx, dive in tqdm(df_csv.iterrows(), total=len(df_csv)):
                    npz_path = os.path.join(p, "dives", f"acoustic_dive_{idx:05d}.npz")
                    npz_data = np.load(npz_path)

                    time = npz_data["time"]
                    precip = dive["precipitation_GPM"]
                    ws = dive["wind_speed"]
                    upwards_mean_15000 = npz_data["spectro"].T[480]
                    upwards_mean_8000 = npz_data["spectro"].T[256]
                    upwards_mean_5000 = npz_data["spectro"].T[160]
                    upwards_mean_2000 = npz_data["spectro"].T[64]

                    for i in range(len(time)):
                        rows.append({
                            "fns": npz_path,
                            "begin_time": time[i],
                            "precipitation_GPM": float(precip),
                            "wind_speed": float(ws),
                            "upwards_mean_15000": upwards_mean_15000[i],
                            "upwards_mean_8000": upwards_mean_8000[i],
                            "upwards_mean_5000": upwards_mean_5000[i],
                            "upwards_mean_2000": upwards_mean_2000[i],
                            "depid": dep
                        })
            return pd.DataFrame(rows)

        elif self.df_data == "npz":
            rows = []
            for p, dep in zip(self.path, self.depid):
                df_csv = pd.read_csv(f"{p}/{dep}_dive.csv")
                for idx, row in tqdm(df_csv.iterrows(), total=len(df_csv)):
                    npz_path = os.path.join(p, dep, "dives", f'acoustic_dive_{int(row["dive"]):05d}.npz')
                    npz = np.load(npz_path)
                    freq, spectro = npz["freq"], npz["spectro"]
                    mask = npz["depth"] > 10
                    spectro_masked = spectro[mask]

                    target_length = 600
                    current_length = spectro_masked.shape[0]
                    if current_length < target_length:
                        pad_width = ((0, target_length - current_length), (0, 0))
                        padded_spectro = np.pad(spectro_masked, pad_width, mode='constant')
                    else:
                        padded_spectro = spectro_masked[:target_length, :]

                    flat_spectro = padded_spectro.flatten()
                    row_npz = np.insert(flat_spectro, 0, row["precipitation_GPM"])
                    rows.append(row_npz)
            final_array = np.stack(rows)
            return pd.DataFrame(final_array)

        else:
            raise Exception("/!\\ df_data must be 'csv', 'npz_reduced' or 'npz'")

class RainClassifier:
    def __init__(self, df: pd.DataFrame, frequency: int = 5000):
        self.df = df
        self.frequency = frequency

    def calculate_and_add_slope(self, freq1, freq2):
        slope_col = f"slope_{freq1}_{freq2}"
        spl1 = self.df[f'upwards_mean_{freq1}']
        spl2 = self.df[f'upwards_mean_{freq2}']
        delta_spl = spl2 - spl1
        delta_log_freq = np.log10(freq2) - np.log10(freq1)
        self.df[slope_col] = delta_spl / delta_log_freq

    def classify_weather(self):
        wind_thresh = 7
        conditions = [
            (self.df["precipitation_GPM"] > 0.1) & (self.df["wind_speed"] < wind_thresh),
            (self.df["precipitation_GPM"] > 0.1) & (self.df["wind_speed"] >= wind_thresh),
        ]
        choices = ["R", "WR"]
        self.df["weather"] = np.select(conditions, choices, default="N")

    def classify_rain(self, offset=1.25, optimised_tresh=False):
        combinations = [("upwards_mean_5000","upwards_mean_15000"),
                        ("upwards_mean_8000","upwards_mean_15000"),
                        ("upwards_mean_8000","slope_8000_15000")]
        coefs = []

        for comb in combinations:
            polydeg = 1
            quant_val = 0.8
            quantile_df = pd.DataFrame({})

            for bin in range(-55, -25, 5):
                _df = self.df[(self.df[comb[0]] > bin) & (self.df[comb[0]] < bin + 5)]
                threshold = _df[comb[1]].quantile(quant_val)
                _df.loc[_df[comb[1]] > threshold, comb[1]] = np.nan
                _df = _df.dropna(subset=[comb[0], comb[1]])
                quantile_df = pd.concat([quantile_df, _df])

            model = np.poly1d(np.polyfit(quantile_df[comb[0]], quantile_df[comb[1]], polydeg))
            a, b = model.coefficients
            coefs.append((comb[0], comb[1], a, b))

        if optimised_tresh:
            conditions = [
                (self.df[coefs[0][1]] > coefs[0][2]*self.df[coefs[0][0]] + coefs[0][3] + offset/2) &
                (self.df[coefs[1][1]] > coefs[1][2]*self.df[coefs[1][0]] + coefs[1][3] + offset) &
                (self.df[coefs[2][1]] > coefs[2][2]*self.df[coefs[2][0]] + coefs[2][3] + offset/4)
            ]
        else:
            conditions = [
                (self.df[coefs[0][1]] > coefs[0][2]*self.df[coefs[0][0]] + coefs[0][3] + offset) &
                (self.df[coefs[1][1]] > coefs[1][2]*self.df[coefs[1][0]] + coefs[1][3] + offset) &
                (self.df[coefs[2][1]] > coefs[2][2]*self.df[coefs[2][0]] + coefs[2][3] + offset)
            ]

        choices = ["R"]
        self.df["Rain_Type_preds"] = np.select(conditions, choices, default="N+WR")

        cond_true = (self.df["precipitation_GPM"] > 0.1) & (self.df["wind_speed"] < 7)
        self.df["Rain_Type"] = np.where(cond_true, "R", "N+WR")

class RainEstimator:
    def __init__(self, df: pd.DataFrame, method_name="Nystuen", frequency: int = 5000, split_method='depid', test_depid='ml17_280a'):
        self.df = df
        self.method_name = method_name
        self.method = empirical_rain[self.method_name]
        self.frequency = frequency
        self.split_method = split_method
        self.test_depid = test_depid

        self.df_r = self.df.loc[self.df["Rain_Type_preds"]=="R"].copy()
        self.df_r = self.df_r.dropna(subset=['precipitation_GPM', f"upwards_mean_{self.frequency}"])
        self.df_r = self.df_r.reset_index(drop=True)

        if self.split_method == 'depid':
            train_split, test_split = get_train_test_split(
                self.df_r.fns.to_numpy(),
                self.df_r.index.to_numpy(),
                self.df_r.depid.to_numpy(),
                method=self.split_method,
                test_depid=self.test_depid
            )
        else:
            train_split, test_split = get_train_test_split(
                self.df_r.fns.to_numpy(),
                self.df_r.index.to_numpy(),
                self.df_r.depid.to_numpy(),
                method=self.split_method
            )
        self.train_split = train_split
        self.test_split = test_split

    def fit(self, func, p0=None):
        xdata = self.df_r.loc[self.train_split, f"upwards_mean_{self.frequency}"].values
        ydata = self.df_r.loc[self.train_split, 'precipitation_GPM'].values
        popt, pcov = curve_fit(func, xdata, ydata, p0=p0)
        self.method['params'] = popt
        return popt

    def predict(self, func):
        xdata = self.df_r.loc[self.test_split, f"upwards_mean_{self.frequency}"].values
        y_pred = func(xdata, *self.method['params'])
        return y_pred

    def score(self, y_pred):
        y_true = self.df_r.loc[self.test_split, 'precipitation_GPM'].values
        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        mae = metrics.mean_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae, "R2": r2}

def get_train_test_split(fns, indices, depid, method='depid', test_depid='ml17_280a'):
    if method == 'depid':
        train_indices = [i for i, d in enumerate(depid) if d != test_depid]
        test_indices = [i for i, d in enumerate(depid) if d == test_depid]
        return train_indices, test_indices
    elif method == 'skf':
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(fns, depid):
            return train_index, test_index
    elif method == 'temporal':
        default = {'split': 0.8, 'scaling_factor': 0.2, 'maxfev': 25000}
        params = {**default}

        n_total = len(indices)
        n_train = int(params['split'] * n_total)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        return train_indices, test_indices

    else:
        raise ValueError(f"Unknown split method: {method}")

class Rain2:
    def __str__(self):
        if not self.estimator:
            print("Estimator not set up yet")
        else :
            for key, value in self.rain_model_stats.items():
                print(f"{key} : {value}")
        return ""

    def __init__(self, depid, path, acoustic_path=[], df_data="csv", frequency=5000):
        self.loader = DataLoader(depid, path, acoustic_path, df_data)
        self.df = self.loader.load()
        self.frequency = frequency
        self.classifier = RainClassifier(self.df, frequency)
        self.estimator = None

        self.rain_model_stats = {}
        self.popt_save = {}

    def preprocess(self):
        self.classifier.calculate_and_add_slope(8000, 15000)
        self.classifier.calculate_and_add_slope(5000, 15000)
        self.classifier.calculate_and_add_slope(2000, 8000)

        self.classifier.classify_weather()
        self.classifier.classify_rain()

    def setup_estimator(self, method_name, split_method='depid', test_depid='ml17_280a'):
        self.estimator = RainEstimator(self.df, method_name=method_name, frequency=self.frequency, split_method=split_method, test_depid=test_depid)

    def fit(self, p0=None, bounds=(-np.inf, np.inf), maxfev=10000):
        if not self.estimator:
            raise RuntimeError("Estimator not set up. Call setup_estimator() first.")

        trainset = self.estimator.df_r.loc[self.estimator.train_split]
        x_train = trainset[f"upwards_mean_{self.frequency}"].to_numpy()
        y_train = trainset['precipitation_GPM'].to_numpy()
        
        print(self.estimator.method)

        func = self.estimator.method['function']
        

        if p0 is None and "parameters" in self.estimator.method:
            p0 = list(self.estimator.method["parameters"].values())

        popt, pcov = curve_fit(
            func,
            x_train,
            y_train,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev
        )

        if "parameters" in self.estimator.method:
            self.estimator.method["parameters"] = dict(zip(self.estimator.method["parameters"].keys(), self.popt))
        
        self.popt_save.update({f'{self.estimator.split_method}_fit' : popt})

    
    def _calculate_metrics(self, y_true, y_pred):
        mae = metrics.mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        r2 = metrics.r2_score(y_true, y_pred)
        var = np.var(np.abs(y_true) - np.abs(y_pred))
        std = np.std(np.abs(y_true) - np.abs(y_pred))
        cc = np.corrcoef(y_true, y_pred)[0, 1]

        self.rain_model_stats.update({
            f'{self.estimator.split_method}_mae': mae,
            f'{self.estimator.split_method}_rmse': rmse,
            f'{self.estimator.split_method}_r2': r2,
            f'{self.estimator.split_method}_var': var,
            f'{self.estimator.split_method}_std': std,
            f'{self.estimator.split_method}_cc': cc
        })

    def predict(self,):
        if not self.estimator:
            raise RuntimeError("Estimator not set up. Call setup_estimator() first.")
        if self.popt is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        testset = self.estimator.df_r.loc[self.estimator.test_split]
        x_test = testset[f"upwards_mean_{self.frequency}"].to_numpy()
        y_test = testset["precipitation_GPM"].to_numpy()
        func = self.estimator.method['function']

        y_pred = func(x_test, *self.popt)
        self._calculate_metrics(y_true=y_test, y_pred=y_pred)
    


