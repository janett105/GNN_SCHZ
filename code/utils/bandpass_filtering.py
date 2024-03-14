from nilearn.image import clean_img
import pandas as pd
import numpy as np

def filtering(subj_id, subjects_dir):
    input_file = f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    output_file = f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_filtered_space-MNI152NLin2009cAsym_desc-preproc_bold.nii'
    mask_file = f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    confounds_file = pd.read_csv(f'{subjects_dir}{subj_id}/func/{subj_id}_rest_confounds.csv').values

    confounds_file =nan_interp(confounds_file)

    filtered_img = clean_img(imgs=input_file,
                            mask_img=mask_file,
                            low_pass=0.1, 
                            high_pass=0.01, 
                            confounds=confounds_file,
                            detrend=False,
                            standardize=False,
                            t_r=2.0)

    filtered_img.to_filename(output_file)

def create_confounds_file(subj_id, subjects_dir):
    error_subjects=[]
    try: 
        input_file = pd.read_csv(f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_desc-confounds_timeseries.tsv', sep = '\t')
        print(input_file)
    except FileNotFoundError as e:  error_subjects.append([{subj_id},e]) 
    print("error subjects : ", error_subjects)
    

def nan_interp(file):
    for col in range(file.shape[1]):
        y = file[:,col]
        nans, x= np.isnan(y), lambda z: z.nonzero()[0]
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        file[:,col] = y
    return file
subjects_dir = 'C:/Users/janet/MyProjects/GNN_SCHZ/data'
#subjects_dir = 'D:/MRI/UCLA_CNP/preprocess/fmriprep/derivatives/'
#filtering('sub-10171', subjects_dir)
create_confounds_file('sub-10171', subjects_dir)