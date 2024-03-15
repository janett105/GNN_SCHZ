import numpy as np

def nan_interp(file):
    """
    input : np.nan이 포함되어 있는 array
    return : nan값 interpolation된 array 
    """
    for col in range(file.shape[1]):
        y = file[:,col]
        nans, x= np.isnan(y), lambda z: z.nonzero()[0]
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        file[:,col] = y
    return file