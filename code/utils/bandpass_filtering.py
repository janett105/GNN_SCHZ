from nilearn.image import clean_img
import pandas as pd
import numpy as np

def filtering(subj_id):
    subjects_dir = 'D:/MRI/UCLA_CNP/preprocess/fmriprep/derivatives/'

    input_file = f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    output_file = f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_filtered_space-MNI152NLin2009cAsym_desc-preproc_bold.nii'
    mask_file = f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    confounds_file = pd.read_csv(f'{subjects_dir}{subj_id}/func/{subj_id}_rest_confounds.csv').values

    np.nan_to_num(confounds_file, copy=False)

    filtered_img = clean_img(imgs=input_file,
                            mask_img=mask_file,
                            low_pass=0.1, 
                            high_pass=0.01, 
                            confounds=confounds_file,
                            t_r=2.0,)

    filtered_img.to_filename(output_file)

# def create_confounds_file():

filtering('sub-10171')