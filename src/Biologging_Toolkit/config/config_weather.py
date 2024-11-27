import numpy as np


def logarithmic(x, a, b, offset):
    return 10 ** (((x - offset) + 10 * a * np.log10(8000)) / (20 * b + 1e-8))


def quadratic(x, a, b, c, offset):
    return a * (x - offset) ** 2 + b * (x - offset) + c


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