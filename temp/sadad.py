import numpy as np
import pandas as pd

subj_id = 'sub-10171'
subjects_dir = 'C:/Users/janet/MyProjects/GNN_SCHZ/data/'
confounds_file = pd.read_csv(f'{subjects_dir}{subj_id}/func/{subj_id}_rest_confounds.csv').values # (152,13)

for col in range(confounds_file.shape[1]):
    y = confounds_file[:,col]
    nans, x= np.isnan(y), lambda z: z.nonzero()[0]
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])