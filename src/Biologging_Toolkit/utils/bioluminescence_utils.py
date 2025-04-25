import numpy as np

def find_sequence(condition, samplerate =  1):
    condition = condition.astype(int)
    edges = np.diff(condition, prepend=0, append=0)
    start_idx = np.where(edges == 1)[0]
    end_idx = np.where(edges == -1)[0] - 1
    length = (end_idx - start_idx + 1) / samplerate
    result = np.vstack([condition[start_idx], start_idx, end_idx, length]).T
    return result