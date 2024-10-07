import numpy as np
from glob import glob
import os 

def db_mean(levels, axis=None):
	"""
	Energetic average of levels.
	:param levels: Sequence of levels.
	:param axis: Axis over which to perform the operation.
	.. math:: L_{mean} = 10 \\log_{10}{\\frac{1}{n}\\sum_{i=0}^n{10^{L/10}}}
	from python-acoustics package
	"""
	levels = np.asanyarray(levels)
	return 10.0 * np.log10((10.0**(levels / 10.0)).mean(axis=axis))


def find_corresponding_file(timestamp, ts):
	"""
	
	"""
	i = 0
	fns, start_time = [], []
	end_time = timestamp.end.iloc[i]
	begin_time = timestamp.begin.iloc[i]
	fn = timestamp.fn.iloc[i]
	for elem in ts:
		if elem > end_time:
			i+=1
			fn = timestamp.fn.iloc[i]
			end_time = timestamp.end.iloc[i]
			begin_time = timestamp.begin.iloc[i]
		fns.append(fn)	
		start_time.append(begin_time)
	return np.array(fns), np.array(start_time)


def find_npz_for_time(ref_times, npz_path):
	"""
    Create an array mapping each time in ds_times to the corresponding npz file.
    
    Parameters:
        ds_times (np.ndarray): Array of epoch times from self.ds['time'].
        npz_files (list): List of (filename, time_array) tuples from npz files.
    
    Returns:
        np.ndarray: Array of the same shape as ds_times, containing the npz filename/index.
	"""
	npz_files = glob.glob(os.path.join(npz_path, 'acoustic*npz'))
    # Initialize an array to store the file index or name for each time in ds_times
	output_array = np.empty(ref_times.shape, dtype=object)  # Store filename/index here
    
    # Prepare npz time ranges
	time_ranges = [(npz_file[0], npz_file[1][0], npz_file[1][-1]) for npz_file in npz_files]
    
	# For each time in ds_times, find the corresponding npz file
	for i, time in np.ndenumerate(ref_times):
		# Perform a binary search over the time_ranges to find the correct npz file
		for npz_file, start_time, end_time in time_ranges:
			if start_time <= time <= end_time:
				output_array[i] = npz_file
				break
                
	return output_array


