import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, make_scorer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import seaborn as sns
import scipy.signal as signal
from sympy import Symbol, expand
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

import matplotlib.cm as cm
from tqdm import tqdm

import sys, os
sys.path.append('../src/')
from Biologging_Toolkit.applications.Rain import Rain

weather4_colors = {
        "R": "#6666ff",
        "WR": "#c27ba0",
        "W": "#28d128",
        "N": "#cccccc"
        }

def load_df(depid, path, feature, ignore_SPL=False):
    
    df = pd.read_csv(path + f'/{depid}/{depid}_dive.csv')
    factor = 1000 if feature.startswith('tp') else 1
    df[feature] = df[feature]*factor
    df['depid']=depid

    if ignore_SPL:
        df = df.dropna(subset=["begin_time", feature])
    else:
        df = df.dropna(subset=["begin_time", "upwards_mean_5000", "upwards_mean_8000", "upwards_mean_2000", "wind_speed", feature,"upwards_mean_15000", "upwards_mean_20000"])
    return df

#-----------------------------------------#---------------------------------#
#       Fonctions des la Biblio           #           ESTIMATION            #
#-----------------------------------------#---------------------------------#

def RE_KKNoufal_2025(df, FKHz=8000, a=14.76, b=44.93, c=5.66, d=15.75, off=0):
    SPL_1_10kHz = df[f'upwards_mean_{FKHz}'] + off
    log_f = np.log10(FKHz)
    log_5 = np.log10(5)
    R = (SPL_1_10kHz + (log_f - log_5) * a - b) / ((log_f - log_5) * c + d)
    return R

def RE_Nystuen_2014(df, a=0.0325, b=1.4416, off=0):
    SPL_5Khz = df['upwards_mean_5000'] + off
    R = (10**(a*SPL_5Khz - b) )
    return R

def RE_Nystuen_1997(df, a=51.9, b=10.6, off=0):
    SPL_5Khz = df['upwards_mean_5000'] + off
    R = (10**((SPL_5Khz - a)/b))
    return R

def RE_Nystuen_2004(df, a=42.5, b=15.4, off=0):
    SPL_5Khz = df['upwards_mean_5000'] + off
    R = (10**((SPL_5Khz - a)/b))
    return R

def RE_Nystuen_8KHz(df, a=42.5, b=15.4, off=0):
    SPL_5Khz = df['upwards_mean_8000'] + off
    R = (10**((SPL_5Khz - a)/b))
    return R

def RE_Nystuen_XKHz(df, a=42.5, b=15.4, off=0, spl="upwards_mean_8000"):
    SPL_XKhz = df[spl] + off
    R = (10**((SPL_XKhz - a)/b))
    return R

def RE_Pensieri_2014(df):
    SPL_5Khz = df['upwards_mean_5000'] + 80
    R = (10**((SPL_5Khz - 64.402)/25))
    return R

def RE_Pickett_1991_LIN2(df, a = -11, b=2.7*10**(-3), c=5.7*10**(-3), off=0):
    SPL_2Khz = df['upwards_mean_2000'] + off
    SPL_8Khz = df['upwards_mean_8000'] + off
    R = a+b*SPL_2Khz + c*SPL_8Khz
    return R

def RE_Pickett_1991_LIN1(df, a = -6.1, b=7.3*10**(-3), off=0):
    SPL_8Khz = df['upwards_mean_8000']+off
    R = a+b*(SPL_8Khz)
    return R

def RE_Pickett_1991_NLIN(df, a=2*10**(-7), off=0, exp=2.2):
    SPL_8Khz = df["upwards_mean_8000"] + off
    R = a*SPL_8Khz**(exp)
    return R

#-----------------------------------------#
#       Objectives pour OPTUNA            #
#-----------------------------------------#
def objective_KKNoufal_2025(trial, df, feature):
    a = trial.suggest_float('a', 10, 20)
    b = trial.suggest_float('b', 30, 60)
    c = trial.suggest_float('c', 1, 10)
    d = trial.suggest_float('d', 10, 25)
    off = trial.suggest_float('off', 0, 100)

    R_est = RE_KKNoufal_2025(df, a=a, b=b, c=c, d=d, off=off)
    return mean_squared_error(df[feature], R_est)

def objective_Nystuen_2014(trial, df, feature):
    a = trial.suggest_float('a', 0.01, 0.05)
    b = trial.suggest_float('b', 1.0, 2.0)
    off = trial.suggest_float('off', 0, 100)

    R_est = RE_Nystuen_2014(df, a=a, b=b, off=off)
    return mean_squared_error(df[feature], R_est)

def objective_Nystuen_1997(trial, df, feature):
    a = trial.suggest_float('a', 40, 60)
    b = trial.suggest_float('b', 5, 20)
    off = trial.suggest_float('off', 0, 100)

    R_est = RE_Nystuen_1997(df, a=a, b=b, off=off)
    return mean_squared_error(df[feature], R_est)

def objective_Nystuen_2004(trial, df, feature):
    a = trial.suggest_float('a', 30, 50)
    b = trial.suggest_float('b', 5, 25)
    off = trial.suggest_float('off', 0, 100)

    R_est = RE_Nystuen_2004(df, a=a, b=b, off=off)
    return mean_squared_error(df[feature], R_est)

def objective_Nystuen_8_KHz(trial, df, feature):
    a = trial.suggest_float('a', -50, 50)
    b = trial.suggest_float('b', 0, 100)

    R_est = RE_Nystuen_8KHz(df, a=a, b=b)
    
    return mean_squared_error(df[feature], R_est)

def objective_Nystuen_X_KHz(trial, df, feature, spl="upwards_mean_8000"):
    a = trial.suggest_float('a', 10, 50)
    b = trial.suggest_float('b', 0, 30)
    off = trial.suggest_float('off', 40, 80)

    R_est = RE_Nystuen_XKHz(df, a=a, b=b, off=off, spl=spl)
    return mean_squared_error(df[feature], R_est)

def objective_Pickett_1991_LIN1(trial, df, feature):
    a = trial.suggest_float('a', -10, 10)
    b = trial.suggest_float('b', 0, 1)
    off = trial.suggest_float('off', 0, 100)

    R_est = RE_Pickett_1991_LIN1(df, a=a, b=b, off=off)
    return mean_squared_error(df[feature], R_est)

def objective_Pickett_1991_NLIN(trial, df, feature):
    a = trial.suggest_float('a', 0, 1)
    exp = trial.suggest_float('exp', 1, 3)
    off = trial.suggest_float('off', 80, 150)

    R_est = RE_Pickett_1991_NLIN(df, a=a, exp=exp, off=off)
    return mean_squared_error(df[feature], R_est)

def objective_Pickett_1991_LIN2(trial, df, feature):
    a = trial.suggest_float('a', -20, 20)
    b = trial.suggest_float('b', 0, 1)
    c = trial.suggest_float('c', 0, 1)
    off = trial.suggest_float('off', 0, 100)

    R_est = RE_Pickett_1991_LIN2(df, a=a, b=b, c=c, off=off)
    return mean_squared_error(df[feature], R_est)

#-----------------------------------------#---------------------------------#
#       Fonctions des la Biblio           #            DETECTION            #
#-----------------------------------------#---------------------------------#

def calculate_background_noise(spectro, percentile=10):
    background_noise = np.percentile(spectro, percentile, axis=1)
    return background_noise

def DE_Zhao_DAsaro_2023(depid, path, dive_number, plot=True, feature="precipitation_GPM", rain_duration=1, return_details = False, SPL_treshold = 4, keepSurface=False, keepOnlySurface=False):
    df = load_df(depid, path, feature, ignore_SPL=True)
    data = np.load(f"{path}{depid}/dives/acoustic_dive_{dive_number:05d}.npz")
    
    if keepSurface:
        if keepOnlySurface :
           mask = data["depth"] < 10
        else : 
            mask = data["depth"] > 0
    else:
        mask = data["depth"] > 10
    spectro = data["spectro"][mask].T
    time = data["time"][mask]
    freq = data["freq"]

    if(len(time)<2):
        return

    background_noise = calculate_background_noise(spectro)
    background_noise_2d = np.tile(background_noise, (spectro.shape[1], 1))

    fluctuation = spectro - background_noise_2d.T

    # dt = np.median(np.diff(time))
    # window_length = int(20 / dt)
    #
    #--
    # time = time[~np.isnan(time)]
    if len(time) < 2:
        raise ValueError("Not enough valid time data points to compute dt.")
    dt = np.median(np.diff(time))
    if np.isnan(dt) or dt == 0:
        raise ValueError(f"Invalid dt value computed: {dt}")
    window_length = int(20 / dt)
    #--

    if window_length % 2 == 0:
        window_length += 1

    minute_scale_fluctuation = signal.medfilt(fluctuation, kernel_size=(1, window_length))
    second_scale_fluctuation = fluctuation - minute_scale_fluctuation

    iteration_rain_duration = int((rain_duration*60)/3)
    mean_spl_list = []
    mean_spl_list_3_9 = []
    mean_spl_list_12_14 = []
    for spl_list in minute_scale_fluctuation.T :
        f5_index = np.where(freq >= 12000)[0]
        f5_16list = spl_list.T[f5_index]

        energies = [10 ** (spl / 10) for spl in f5_16list]
        energie_moyenne = np.mean(energies)
        spl_moyen = 10 * np.log10(energie_moyenne)
        mean_spl_list.append(spl_moyen)

        f3_index = np.where((freq <= 9000) & (freq >= 8000))[0]
        f3_9list = spl_list.T[f3_index]

        energies = [10 ** (spl / 10) for spl in f3_9list]
        energie_moyenne = np.mean(energies)
        spl_moyen = 10 * np.log10(energie_moyenne)
        mean_spl_list_3_9.append(spl_moyen)

        f12_index = np.where((freq <= 14000) & (freq >= 12000))[0]
        f12_14list = spl_list.T[f12_index]

        energies = [10 ** (spl / 10) for spl in f12_14list]
        energie_moyenne = np.mean(energies)
        spl_moyen = 10 * np.log10(energie_moyenne)
        mean_spl_list_12_14.append(spl_moyen)

    flag = False
    for i in range(len(mean_spl_list) - (iteration_rain_duration-1)): 
        if all(spl > SPL_treshold for spl in mean_spl_list[i:i+iteration_rain_duration]):
            flag = True
            break

    if(plot):
        vmins, vmaxs = np.min(second_scale_fluctuation), np.max(second_scale_fluctuation)
        vminm, vmaxm = np.min(minute_scale_fluctuation), np.max(minute_scale_fluctuation)

        plt.figure(figsize=(15, 10))
        plt.subplot(4,1,1)
        plt.imshow(spectro, origin="lower", aspect='auto', cmap='seismic', interpolation='none',
                extent=[(time.min()-time.min())/60, (time.max()-time.min())/60, freq.min(), freq.max()], vmin=-100, vmax=50)
        plt.colorbar(label='Intensity')
        plt.title(f'Spectrogramme:{depid}-dive n°{dive_number}\n{round(df.iloc[dive_number][feature],2)}mm/h\n{round(df.iloc[dive_number]["wind_speed"],2)}m/s')

        plt.subplot(4,1,2)
        plt.imshow(background_noise_2d.T, origin="lower", aspect='auto', cmap='seismic', interpolation='none',
                extent=[(time.min()-time.min())/60, (time.max()-time.min())/60, freq.min(), freq.max()], vmin=-100, vmax=50)
        plt.colorbar(label='Intensity')
        plt.title('Bruit de fond')

        plt.subplot(4,1,3)
        plt.imshow(minute_scale_fluctuation, origin="lower", aspect='auto', cmap='seismic', interpolation='none',
                extent=[(time.min()-time.min())/60, (time.max()-time.min())/60, freq.min(), freq.max()], vmin=vminm, vmax=vmaxm)
        plt.colorbar(label='Intensity')
        plt.title('Fluctuation (T>20s)')
        plt.xlabel('Temps (minutes)')
        plt.ylabel('Fréquence (Hz)')

        plt.subplot(4,1,4)
        plt.imshow(second_scale_fluctuation, origin="lower", aspect='auto', cmap='seismic', interpolation='none',
                extent=[(time.min()-time.min())/60, (time.max()-time.min())/60, freq.min(), freq.max()], vmin=vmins, vmax=vmaxs)
        plt.colorbar(label='Intensity')
        plt.title('Fluctuation (T<20s)')
        plt.xlabel('Temps (minutes)')
        plt.ylabel('Fréquence (Hz)')
        plt.tight_layout()
        plt.show()
    if (not return_details) :
        return "R" if flag==True else "N"
    else :
        output_dict = {"spectro":spectro,
                       "freq":freq, 
                       "time":time, 
                       "background_noise_2d":background_noise_2d,
                       "second_scale_fluctuation":second_scale_fluctuation,
                       "minute_scale_fluctuation":minute_scale_fluctuation,
                       "mean_spl_list": mean_spl_list, 
                       "flag":"R" if flag==True else "N",
                       "mean_spl_list_low": mean_spl_list_3_9,
                       "mean_spl_list_high": mean_spl_list_12_14,}
        return output_dict

def DE_Custom_Nystuen_2015(df, a=-32,b=-36,c=0.7,d=-60,e=2,f=-10):
    spl5 =df['upwards_mean_5000']
    spl8 =df['upwards_mean_8000']
    spl15 =df['upwards_mean_15000']
    slope8_15 = df['slope_8000_15000']
    conditions = [
        (spl5<a) & (spl8<b) & (spl15>c*spl8+d) & (slope8_15>e*spl8+f)
    ]
    df['Rain_Type_Preds'] = np.select(conditions, ["R"], default='N')
    
    # for row in df.iterrows():
    #     row = row[1]
    #     if row["Rain_Type"]=="R":
    #         spl5 = round(row['upwards_mean_5000'],1)
    #         spl8 = round(row['upwards_mean_8000'],1)
    #         spl15 = round(row['upwards_mean_15000'],1)
    #         slope8_15 = round(row['slope_8000_15000'],1)
    #         print(f"(spl5<a) & (spl8<b) & (spl15>c*spl8+d) & (slope8_15>e*spl8+f)")
    #         print(f"({spl15}<{a}) & ({spl8}<{b}) & ({spl15}>{c*spl8+d}) & ({slope8_15}>{e*spl8+f})")
    #         print(f"({spl15}<{a}) & ({spl8}<{b}) & ({spl15}>{c}*{spl8}+{d}) & ({slope8_15}>{e}*{spl8}+{f})\n")
    # print(f"a={a} | b={b} | c={c} | d={d} | e={e} | f={f}")
    return df

def rain_only_accuracy_score(y_true, y_pred, label="R"):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true == label
    if mask.sum() == 0:
        return 0.0
    return (y_true[mask] == y_pred[mask]).mean()

def weighted_rain_accuracy_score(y_true, y_pred, label_r="R", label_n="N", weight_r=0.75, weight_n=0.25):
    import numpy as np

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask_r = y_true == label_r
    if mask_r.sum() == 0:
        accuracy_r = 0.0
    else:
        accuracy_r = (y_true[mask_r] == y_pred[mask_r]).mean()

    mask_n = y_true == label_n
    if mask_n.sum() == 0:
        accuracy_n = 0.0
    else:
        accuracy_n = (y_true[mask_n] == y_pred[mask_n]).mean()

    weighted_accuracy = (accuracy_r * weight_r) + (accuracy_n * weight_n)
    return weighted_accuracy

def objective_DE_Custom_Nystuen_2015(trial, df, metrique = "weighted_rain"):
    params = {
        "a": trial.suggest_float("a", -35, -30),
        "b": trial.suggest_float("b", -40, -35),
        "c": trial.suggest_float("c", .5, 1.5),
        "d": trial.suggest_float("d", -70, -50),
        "e": trial.suggest_float("e", 1, 3),
        "f": trial.suggest_float("f", -20, 20),
    }

    df_pred = DE_Custom_Nystuen_2015(df.copy(), **params)

    if(metrique=="weighted_rain"):
        score = weighted_rain_accuracy_score(df_pred["Rain_Type"], df_pred["Rain_Type_Preds"])
    elif(metrique=="rain_only"):
        score = rain_only_accuracy_score(df_pred["Rain_Type"], df_pred["Rain_Type_Preds"])
    elif(metrique=="f1_score"):
        score = f1_score(df_pred["Rain_Type"], df_pred["Rain_Type_Preds"], average="weighted")
    elif(metrique=="accuracy_score"):
        score = accuracy_score(df_pred["Rain_Type"], df_pred["Rain_Type_Preds"])
    return score

def DE_Ma_Nystuen_2005(df, cond_1=194, cond_2=2.35, cond_3=48, cond_4=53):
    df['upwards_mean_5000']
    df['upwards_mean_20000']
    conditions = [
        (df['upwards_mean_20000'] > cond_1 - cond_2 * df['upwards_mean_5000']) |
        ((df['upwards_mean_20000'] > cond_3) & (df['upwards_mean_5000'] > cond_4))
    ]
    df['Rain_Type_Preds'] = np.select(conditions, ["Rain"], default='No-Rain')
    return df

def objective_DE_Ma_Nystuen_2005(trial, df):
    params = {
        "cond_1": trial.suggest_float("cond_1", 150, 250),
        "cond_2": trial.suggest_float("cond_2", 1.0, 4.0),
        "cond_3": trial.suggest_float("cond_3", 30, 70),
        "cond_4": trial.suggest_float("cond_4", 30, 70),
    }

    df_pred = DE_Ma_Nystuen_2005(df.copy(), **params)

    score = f1_score(df_pred["Rain_Type"], df_pred["Rain_Type_Preds"], average="weighted")
    return score

def DE_Nystuen_2014(
    df,
    offset20=0,
    offset15=0,
    offset8=0,
    offset5=0,
    cond1_slope=0.75,
    cond1_offset=5,
    cond1_max5000=70,
    cond2_min8000=60,
    cond2_min_slope_2_8=-13,
    cond2_min20000=45,
    cond3_max8000=50,
    cond3_min_slope_8_15=-5,
    cond3_min20000=35,
    cond3_ratio8000=0.9,
    cond4_quad1=-0.1144,
    cond4_lin1=12.728,
    cond4_const1=-307,
    cond4_quad2=-0.1,
    cond4_lin2=11.5,
    cond4_const2=-281,
    cond4_min8000=51,
    cond4_max8000=64,
    cond4_min_slope_2_8=-13
):
    df = df.copy()
    df['upwards_mean_20000'] += offset20
    df['upwards_mean_5000'] += offset5
    df['upwards_mean_8000'] += offset8
    df['upwards_mean_15000'] += offset15

    conditions = [
        # Medium rain (stratiform)
        (df['upwards_mean_20000'] > df['upwards_mean_5000'] * cond1_slope + cond1_offset) &
        (df['upwards_mean_5000'] <= cond1_max5000),

        # Heavy rain (convective)
        (df['upwards_mean_8000'] > cond2_min8000) &
        (df['slope_2_8'] > cond2_min_slope_2_8) &
        (df['upwards_mean_20000'] > cond2_min20000),

        # Drizzle
        (df['upwards_mean_8000'] < cond3_max8000) &
        (df['slope_8_15'] > cond3_min_slope_8_15) &
        (df['upwards_mean_20000'] > cond3_min20000) &
        (df['upwards_mean_20000'] > df['upwards_mean_8000'] * cond3_ratio8000),

        # Rain with high winds
        (df['upwards_mean_20000'] > (cond4_quad1 * df['upwards_mean_8000']**2 + cond4_lin1 * df['upwards_mean_8000'] + cond4_const1)) &
        (df['upwards_mean_20000'] < (cond4_quad2 * df['upwards_mean_8000']**2 + cond4_lin2 * df['upwards_mean_8000'] + cond4_const2)) &
        (df['upwards_mean_8000'] > cond4_min8000) &
        (df['upwards_mean_8000'] < cond4_max8000) &
        (df['slope_2_8'] > cond4_min_slope_2_8)
    ]

    choices = ['MR', 'HR', 'D', 'RW']
    df['Rain_Type_Preds'] = np.select(conditions, choices, default='None')
    return df

def objective_DE_Nystuen_2014(trial, df):
    
    # params = {
    #     "offset20": trial.suggest_float("offset20", 0, 100),
    #     "offset15": trial.suggest_float("offset15", 0, 100),
    #     "offset8": trial.suggest_float("offset8", 0, 100),
    #     "offset5": trial.suggest_float("offset5", 0, 100),
    #     "cond1_slope": trial.suggest_float("cond1_slope", 0.5, 1.0),
    #     "cond1_offset": trial.suggest_float("cond1_offset", 0, 10),
    #     "cond1_max5000": trial.suggest_float("cond1_max5000", 60, 80),
    #     "cond2_min8000": trial.suggest_float("cond2_min8000", 50, 70),
    #     "cond2_min_slope_2_8": trial.suggest_float("cond2_min_slope_2_8", -20, -5),
    #     "cond2_min20000": trial.suggest_float("cond2_min20000", 30, 60),
    #     "cond3_max8000": trial.suggest_float("cond3_max8000", 40, 60),
    #     "cond3_min_slope_8_15": trial.suggest_float("cond3_min_slope_8_15", -10, 0),
    #     "cond3_min20000": trial.suggest_float("cond3_min20000", 30, 50),
    #     "cond3_ratio8000": trial.suggest_float("cond3_ratio8000", 0.7, 1.0),
    #     "cond4_quad1": trial.suggest_float("cond4_quad1", -0.2, -0.05),
    #     "cond4_lin1": trial.suggest_float("cond4_lin1", 10, 15),
    #     "cond4_const1": trial.suggest_float("cond4_const1", -350, -250),
    #     "cond4_quad2": trial.suggest_float("cond4_quad2", -0.2, -0.05),
    #     "cond4_lin2": trial.suggest_float("cond4_lin2", 10, 15),
    #     "cond4_const2": trial.suggest_float("cond4_const2", -350, -250),
    #     "cond4_min8000": trial.suggest_float("cond4_min8000", 45, 55),
    #     "cond4_max8000": trial.suggest_float("cond4_max8000", 60, 70),
    #     "cond4_min_slope_2_8": trial.suggest_float("cond4_min_slope_2_8", -20, -5),
    # }
    params = {
    "cond1_slope": trial.suggest_float("cond1_slope", 0, 3.0),
    "cond1_offset": trial.suggest_float("cond1_offset", -10, 30),
    "cond1_max5000": trial.suggest_float("cond1_max5000", 40, 120),
    "cond2_min8000": trial.suggest_float("cond2_min8000", 30, 90),
    "cond2_min_slope_2_8": trial.suggest_float("cond2_min_slope_2_8", -40, 10),
    "cond2_min20000": trial.suggest_float("cond2_min20000", 10, 80),
    "cond3_max8000": trial.suggest_float("cond3_max8000", 20, 80),
    "cond3_min_slope_8_15": trial.suggest_float("cond3_min_slope_8_15", -30, 20),
    "cond3_min20000": trial.suggest_float("cond3_min20000", 10, 70),
    "cond3_ratio8000": trial.suggest_float("cond3_ratio8000", 0.4, 2.0),
    "cond4_quad1": trial.suggest_float("cond4_quad1", -0.5, 0.1),
    "cond4_lin1": trial.suggest_float("cond4_lin1", 0, 30),
    "cond4_const1": trial.suggest_float("cond4_const1", -500, -150),
    "cond4_quad2": trial.suggest_float("cond4_quad2", -0.5, 0.1),
    "cond4_lin2": trial.suggest_float("cond4_lin2", 0, 30),
    "cond4_const2": trial.suggest_float("cond4_const2", -500, -150),
    "cond4_min8000": trial.suggest_float("cond4_min8000", 30, 70),
    "cond4_max8000": trial.suggest_float("cond4_max8000", 50, 85),
    "cond4_min_slope_2_8": trial.suggest_float("cond4_min_slope_2_8", -40, 10),
    }

    df_pred = DE_Nystuen_2014(df, **params)
    return 1.0 - f1_score(df["Rain_Type"], df_pred["Rain_Type_Preds"], average="weighted")  

#---------------------------#
#           PLOT            #
#---------------------------#

def plot_estimation_vs_ground_truth(df, depid, feature, rain_est):
    dates = [datetime.fromtimestamp(ts) for ts in df['begin_time']]
    plt.figure(figsize=(8,5))
    plt.plot(dates, df[feature], label=f"Rain {"ERA5" if feature.startswith("tp") else "GPM"}")
    plt.plot(dates, rain_est, label=f"Rain Estimation")

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Un marqueur au début de chaque mois
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[5,10,15,20,25,30]))  # Marqueurs pour les 8, 15, 22
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Nom du mois
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%d'))  # Numéro du jour

    # Afficher les étiquettes des mois et des jours
    plt.gca().xaxis.set_tick_params(which='major', labelsize=10, rotation=0)
    plt.gca().xaxis.set_tick_params(which='minor', labelsize=8)
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gcf().autofmt_xdate()

    plt.ylabel("precipitation (mm/h)")
    plt.xlabel("time")
    plt.title(f'{depid} estimation fit', fontsize=12, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_spl_rain(df):
    colors = {
        "R": "#6666ff",
        "WR": "#c27ba0",
        # "W": "#28d128",
        "N": "#cccccc"
        }
    combs = [("upwards_mean_5000","upwards_mean_15000"),
            ("upwards_mean_8000","upwards_mean_15000"),
        ("upwards_mean_8000","slope_2000_8000"),
            ("upwards_mean_8000","slope_8000_15000")]

    i=0
    fig, ax = plt.subplots(1,4,figsize=(16, 3.5))
    for comb in combs:
        for weather_type, color in colors.items():
            subset = df[df['weather'] == weather_type]
            ax[i].scatter(
                subset[comb[0]],
                subset[comb[1]],
                c=color,
                alpha=0.5,
                label=weather_type
            )

        ax[i].set_xlabel(comb[0])
        ax[i].set_ylabel(comb[1])
        ax[i].legend(title="Weather")
        i+=1
    plt.tight_layout()
    plt.show()

def plot_spl_by_rain_type(df, freqs, precip_value, depid, fill=True):
    """
    Trace les courbes SPL moyennes par fréquence pour chaque type de pluie défini par 'Rain_Type'.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        freqs (list): Liste des fréquences à considérer.
        precip_value (str): Nom de la colonne contenant les valeurs de précipitation.
        depid (str): Identifiant de la station ou du capteur, utilisé dans le titre.
    """
    rain_types = ["N", "W", "WR", "R"]
    spl_by_rain_type = {rtype: [] for rtype in rain_types}
    colors = {
        "R": "#6666ff",
        "WR": "#c27ba0",
        "W": "#28d128",
        "N": "#cccccc"
    }

    for _, row in df.iterrows():
        rtype = row["Rain_Type"]
        spl_row = []

        for freq in freqs:
            spl_row.append(row.get(f"upwards_mean_{freq}", np.nan))
        
        spl_by_rain_type[rtype].append(spl_row)

    plt.figure(figsize=(10, 6))
    for rtype, values in spl_by_rain_type.items():
        if len(values) == 0:
            continue
        values = np.array(values)
        mean_spl = np.nanmean(values, axis=0)
        std_spl = np.nanstd(values, axis=0)
        plt.plot(freqs, mean_spl, marker='o', alpha=0.6, label=f"Type {rtype}",color=colors[rtype])
        if fill:
            plt.fill_between(freqs, mean_spl - std_spl, mean_spl + std_spl, alpha=0.5,color=colors[rtype])

    plt.legend(loc='upper right')
    plt.title(f"{depid}: SPL moyen par fréquence selon la météo", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("SPL (dB re 1 µPa)")
    plt.tight_layout()
    plt.show()

def plot_WindRain_scatter(depid, path, estimated_rain, var_name, palette ="viridis"):
    df = pd.read_csv(path + f'/{depid}/{depid}_dive.csv')
    feature = "precipitation_GPM"
    factor = 1000 if feature.startswith('tp') else 1
    df = df.dropna(subset=["begin_time", "upwards_mean_5000", "upwards_mean_8000", "upwards_mean_2000", "wind_speed", feature])
    df["dives"] = "all"

    df["estimated_rain"] = estimated_rain
    df['wind speed category (m/s)'] = pd.cut(df['wind_speed'], bins=5, precision=0)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": True  
    })

    g = sns.jointplot(
        data=df,
        x="wind_speed",
        y="estimated_rain",
        hue='wind speed category (m/s)', 
        kind="scatter",
        palette=palette,
        alpha=0.8,
        # height=10,  # Augmente la hauteur de la figure globale
        ratio=3,
        marginal_kws=dict(
            fill=True,
            bw_adjust=0.3,
            alpha=0.8,
            log_scale=True
        ),
    )
    g.ax_marg_x.set_xscale("linear")
    g.ax_joint.set_xscale("linear")

    g.set_axis_labels("Wind Speed (m/s)", "Estimated Precipitation (mm/h)", fontsize=12)

    g.fig.text(0.45, -0.05, f'Précipitations estimées ({var_name}) (mm/h) en fonction \ndu vent (m/s) pour chaque plongée', ha='center', fontsize=14)
    plt.show()
    
def plot_RainGPMRain_scatter(depid, path, estimated_rain, var_name, palette ="viridis"):
    df = pd.read_csv(path + f'/{depid}/{depid}_dive.csv')
    feature = "precipitation_GPM"
    factor = 1000 if feature.startswith('tp') else 1
    df = df.dropna(subset=["begin_time", "upwards_mean_5000", "upwards_mean_8000", "upwards_mean_2000", "wind_speed", feature])
    df["dives"] = "all"

    df["estimated_rain"] = estimated_rain
    df['wind_speed_category'] = pd.cut(df['wind_speed'], bins=5, precision=0)
    df = df[df['precipitation_GPM'] > 0]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": True  
    })
    g = sns.jointplot(
        data=df,
        x="precipitation_GPM",
        y="estimated_rain",
        hue="wind_speed_category", 
        kind="scatter",
        palette=palette,
        alpha=0.8,
        marginal_kws=dict(
            fill=True,
            bw_adjust=0.2,
            alpha=0.8,
            log_scale=True,
        )
    )


    lims = [
        np.min([df["precipitation_GPM"].min(), estimated_rain.min()]),
        np.max([df["precipitation_GPM"].max(), estimated_rain.max()])
    ]
    g.ax_joint.plot(lims, lims, '--', color='black')

    g.set_axis_labels("Precipitation GPM (mm/h)", "Estimated Precipitation (mm/h)", fontsize=12,)
    g.fig.text(0.45, -0.05, f'Précipitations estimées ({var_name}) (mm/h) en fonction \ndes précipitations GPM (mm/h) pour chaque plongée', ha='center', fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_RainSourcesDifferences_scatter(depid,path, source1="tpmaxPool", source2="precipitation_GPM", palette ="viridis", var_name1="ERA5", var_name2="GPM", useLog = True, reduce0=False):
    df = pd.read_csv(path + f'/{depid}/{depid}_dive.csv')
    factor1 = 1000 if source1.startswith('tp') else 1
    factor2 = 1000 if source2.startswith('tp') else 1
    df[source1] = df[source1]*factor1
    df[source2] = df[source2]*factor2
    
    df = df.dropna(subset=["begin_time", "wind_speed", source2, source1])
    df = df[(df[source1] > 1e-1) & (df[source2] > 1e-1)]
    # df = df[(df[source1] > 1e-4) & (df[source2] > 1e-4)]
    df["dives"] = "all"

    df = df[df['precipitation_GPM'] > 0]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": True  
    })
    if useLog == True :
        g = sns.jointplot(
        data=df,
        x=source1,
        y=source2,
        hue="dives", 
        kind="scatter",
        palette=palette,
        alpha=0.8,
        marginal_kws=dict(
            fill=True,
            bw_adjust=0.2,
            alpha=0.8,
            log_scale=useLog,
        )
    )
    else :
        g = sns.jointplot(
        data=df,
        x=source1,
        y=source2,
        kind="kde",
        cmap = palette,
        alpha=0.8,
        fill=False,
        marginal_kws=dict(
            fill=True,
            bw_adjust=0.2,
            alpha=0.8,
            log_scale=useLog,
            color="rebeccapurple"#'#9c78a4'
        ))


    lims = [
        np.min([df[source2].min(), df[source1].min()]),
        np.max([df[source2].max(), df[source1].max()])
    ]

    g.ax_joint.plot([[0,0],[7,7]], [[0,0],[7,7]], '--', color='black')
    
    sns.regplot(
        data=df,
        x=source1,
        y=source2,
        scatter=False,
        ax=g.ax_joint,
        line_kws={"color": "red", "linestyle": "-", "linewidth": 2}
    )

    g.set_axis_labels(f"{var_name1} (mm/h)", f"{var_name2} (mm/h)", fontsize=12,)
    g.fig.text(0.45, -0.05, f'Comparaison des données de précipitations (mm/h) entre \n {var_name1} et {var_name2} pour le dataset {depid}', ha='center', fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_classification_spl(df):
    df = df.dropna(subset=["Rain_Type_preds", "Rain_Type"]).copy()

    color_map = {"R": "royalblue", "N+WR": "lightgrey"}
    combs = [
        ("upwards_mean_5000", "upwards_mean_15000"),
        ("upwards_mean_8000", "upwards_mean_15000"),
        ("upwards_mean_8000", "slope_2000_8000"),
        ("upwards_mean_8000", "slope_8000_15000")
    ]

    # Préparer les sous-groupes
    groups = {
        "True Positives": df[df["Rain_Type_preds"] == "R"],
        "True Negatives": df[df["Rain_Type_preds"] != "R"],
        "False Positives": df[(df["Rain_Type_preds"] == "R") & (df["Rain_Type"] != "R")],
        "False Negatives": df[(df["Rain_Type_preds"] != "R") & (df["Rain_Type"] == "R")]
    }

    colors = {
        "True Positives": "royalblue",
        "True Negatives": "lightgrey",
        "False Positives": "orange",
        "False Negatives": "red"
    }

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.1))

    for i, (x, y) in enumerate(combs):
        for label, data in groups.items():
            ax[i].scatter(
                data[x],
                data[y],
                c=colors[label],
                alpha=0.5,
                label=label
            )
        ax[i].set_xlabel(x)
        ax[i].set_ylabel(y)

    # Extraire légende depuis le premier subplot
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_classification_histogram(df):
    df_r_tp = df[(df["Rain_Type"]=="R") & (df["Rain_Type_preds"]=="R")].copy()
    df_r_tn = df[(df["Rain_Type"]=="N+WR") & (df["Rain_Type_preds"]=="N+WR")].copy()
    df_r_fp = df[(df["Rain_Type"]=="N+WR") & (df["Rain_Type_preds"]=="R")].copy()
    df_r_fn = df[(df["Rain_Type"]=="R") & (df["Rain_Type_preds"]=="N+WR")].copy()

    sns.histplot(data=df_r_tp, x="precipitation_GPM", fill=True, alpha=0.6, label='True Positives', color="darkgray", kde=False, edgecolor=None)
    sns.histplot(data=df_r_fn, x="precipitation_GPM", fill=True, alpha=0.6, label='False Negatives', color="red", kde=False, edgecolor=None)
    # sns.histplot(data=df_r_tn, x="precipitation_GPM", fill=True, alpha=0.0006, label='True Negatives', color="darkorange", kde=True, edgecolor=None)
    # sns.histplot(data=df_r_fp, x="precipitation_GPM", fill=True, alpha=0.6, label='False Positives', color = "royalblue", kde=True, edgecolor=None)
    
    leg = plt.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    plt.xlim(0,8)
    plt.grid(True, linestyle="--", alpha=0.3)

def plot_weather_SPL_regression(df, x = "upwards_mean_8000", y = "slope_8000_15000", height=5) :
    df = df.reset_index(drop=True).copy()
    polydeg = 1
    quant_val = 0.8

    quantile_df = pd.DataFrame({})
    for bin in range(-55, -25, 5):
        _df = df[(df[x] > bin) & (df[x] < bin + 5)]
        threshold = _df[y].quantile(quant_val)
        _df.loc[_df[y] > threshold, y] = np.nan
        _df = _df.dropna(subset=[x, y])
        quantile_df = pd.concat([quantile_df, _df])

    model = np.poly1d(np.polyfit(quantile_df[x], quantile_df[y], polydeg))
    x_Symb = Symbol(x)

    palette = {
        'N': 'gray',
        'WR': 'purple',
        'R': 'blue',
        # 'W': 'green'
    }

    g = sns.JointGrid(data=df, x=x, y=y, hue='weather', palette=weather4_colors, height=height)

    for weather_type, color in palette.items():
        if weather_type != "W" :
            subset = df[df['weather'] == weather_type]
            g.ax_joint.scatter(subset[x], subset[y], c=color, alpha=0.4 if weather_type=='N' else 0.6, label=weather_type)

    x_vals = np.linspace(df[x].min(), df[x].max(), 100)
    y_vals = model(x_vals)
    g.ax_joint.plot(x_vals, y_vals, ':', color="black", linewidth=2)

    for weather_type, color in palette.items():
        subset = df[df['weather'] == weather_type]
        sns.kdeplot(data=subset, y=y, ax=g.ax_marg_y, fill=True, alpha=0.6, color=color)

    g.ax_joint.set_xlabel(x, fontsize=14)
    g.ax_joint.set_ylabel(y, fontsize=14)
    g.ax_joint.legend(title="Weather")


    plt.show()
    print(f"{y} = {expand(model(x_Symb))}")
    a, b = model.coefficients
    print(f"y = {a} * x + {b}")

def plot_confusion_matrix(df):
    conditions = [
        (df["precipitation_GPM"] > 0.1) & (df["wind_speed"] < 7)
    ]
    choices = ["R"]
    df["Rain_Type"] = np.select(conditions, choices, default="N+WR")

    cm = confusion_matrix(df["Rain_Type"], df["Rain_Type_preds"], normalize="true")


    labels = df["Rain_Type"].unique()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)#, fmt='d' )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def plot_rain_estimation_cumulated(inst:Rain, subset = "test"):
    
    if subset == "test" :
        time = pd.to_datetime(inst.df_r["begin_time"].loc[inst.test_split], unit='s')
        gt_values = inst.df_r["precipitation_GPM"].loc[inst.test_split].values

    elif subset == "train":
        time = pd.to_datetime(inst.df_r["begin_time"].loc[inst.train_split], unit='s')
        gt_values = inst.df_r["precipitation_GPM"].loc[inst.train_split].values

    else :
        time = pd.to_datetime(inst.df_r["begin_time"], unit='s')
        gt_values = inst.df_r["precipitation_GPM"].values
    

    gt_cumulated = np.cumsum(gt_values)
    plt.plot(time, gt_cumulated, label="IMERG (GPM NASA)")

    for split_rule in inst.popt :
        a, b = inst.popt[split_rule]
        est = inst.method["function"](inst.df_r["upwards_mean_5000"], a, b)
        est_values = est.loc[inst.test_split].values if subset=="test" else est.loc[inst.train_split].values if subset=='train' else est.values
        est_cumulated = np.cumsum(est_values)
        plt.plot(time, est_cumulated,label=f'Estimation {inst.method_name} (split:{split_rule})')
        
        
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d%b'))
    plt.ylabel("Cumulated precipitation (mm/h)")
    plt.legend()

def plot_rain_estimation(inst:Rain, subset = "test"):

    if subset == "test" :
        time = pd.to_datetime(inst.df_r["begin_time"].loc[inst.test_split], unit='s')
        plt.plot(time,inst.df_r["precipitation_GPM"].loc[inst.test_split], label="IMERG (GPM NASA)")
        
    elif subset == "train":
        time = pd.to_datetime(inst.df_r["begin_time"].loc[inst.train_split], unit='s')
        plt.plot(time,inst.df_r["precipitation_GPM"].loc[inst.train_split], label="IMERG (GPM NASA)")

    else :
        time = pd.to_datetime(inst.df_r["begin_time"], unit='s')
        plt.plot(time,inst.df_r["precipitation_GPM"], label="IMERG (GPM NASA)")

    for split_rule in inst.popt :
        a, b = inst.popt[split_rule]
        est = inst.method["function"](inst.df_r["upwards_mean_5000"], a, b)
        est_values = est.loc[inst.test_split].values if subset=="test" else est.loc[inst.train_split].values if subset=='train' else est.values
        plt.plot(time,est_values, label=f'Estimation {inst.method_name} (split:{split_rule})')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d%b'))
    plt.ylabel("Precipitation (mm/h)")
    plt.legend()

def plot_spectro_size_repartition(depids,path):
    dims = []
    for depid in depids:
        _df = pd.read_csv(os.path.join(path,depid,f"{depid}_dive.csv"))
        for _, row in tqdm(_df.iterrows(),desc=depid):
            npz_path = os.path.join(path,depid,"dives",f'acoustic_dive_{int(row["dive"]):05d}.npz')
            npz = np.load(npz_path)
            freq, spectro = npz["freq"], npz["spectro"]
            mask = npz["depth"] > 10
            spectro_masked = spectro[mask]

            dims.append(((spectro_masked.shape),depid))

    plt.figure(figsize=(15, 5))
    colors = cm.tab20(np.linspace(0, 1, len(depids)))

    data_grouped = []
    labels = []

    for i, depid in enumerate(depids):
        values = [dims[j][0][0] for j, d in enumerate(dims) if d[1] == depid]
        data_grouped.append(values)
        labels.append(depid)

    box = plt.boxplot(data_grouped, patch_artist=True, labels=labels)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def print_metrics(y_preds, y_test):
    mae = metrics.mean_absolute_error(y_test, y_preds)
    rmse = metrics.root_mean_squared_error(y_test, y_preds)
    r2 = metrics.r2_score(y_test, y_preds)
    var = np.var(abs(y_test)-abs(y_preds))
    std = np.std(abs(y_test)-abs(y_preds))
    cc = np.corrcoef(y_test, y_preds)[0][1]
    print(f"mae : {mae}")
    print(f"rmse : {rmse}")
    print(f"r2 : {r2}")
    print(f"var : {var}")
    print(f"std : {std}")
    print(f"cc : {cc}")