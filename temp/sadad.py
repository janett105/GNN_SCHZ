import numpy as np


y= np.array([1, 1, 1, np.nan, NaN, 2, 2, NaN, 0])
nans, x= np.isnan(y), lambda z: z.nonzero()[0]
y[nans]= np.interp(x(nans), x(~nans), y[~nans])