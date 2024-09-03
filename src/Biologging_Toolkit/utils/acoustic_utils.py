import numpy as np


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




