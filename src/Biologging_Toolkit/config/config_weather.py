import numpy as np


def logarithmic(x, a, b, offset):
    return 10 ** (((x - offset) + 10 * a * np.log10(8000)) / (20 * b + 1e-8))


def quadratic(x, a, b, c, offset):
    return a * (x - offset) ** 2 + b * (x - offset) + c

def logRR(x, a, b):
    return 10**((x + a)/b)

def logRRoffset(x,a,b,offset):
    return 10**((x - a + offset)/b)

empirical = {
    "Hildebrand": {
        "frequency": 8000,
        "function": logarithmic,
        "averaging_duration": 3600,
        "parameters": {"a": 78, "b": 1.5},
    },
    "Pensieri": {
        "frequency": 8000,
        "function": quadratic,
        "averaging_duration": 4.5,
        "parameters": {"a": 0.044642, "b": -3.2917, "c": 63.016},
    },
}

empirical_rain = {
    "Nystuen" : {
        "frequency": "upwards_mean_5000",
        "function" : logRR,
        "averaging_duration": 4.5,
        "parameters": {"a": 51.9, "b": 10.6},
    },
    "Nystuen1997" : {
        "frequency": "upwards_mean_5000",
        "function" : logRRoffset,
        "averaging_duration": 4.5,
        "parameters": {"a": 51.9, "b": 10.6},
    },
    "Nystuen2004" : {
        "frequency": "upwards_mean_5000",
        "function" : logRRoffset,
        "averaging_duration": 4.5,
        "parameters": {"a": 42.5, "b": 15.2},
    },
    "Pensieri2015" : {
        "frequency": "upwards_mean_5000",
        "function" : logRRoffset,
        "averaging_duration": 4.5,
        "parameters": {"a": 64.4, "b": 25},
    },
    "Nystuen2015" : {
        "frequency": "upwards_mean_5000",
        "function" : logRRoffset,
        "averaging_duration": 4.5,
        "parameters": {"a": 44.35, "b": 30.77},
    }
}