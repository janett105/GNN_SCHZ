from nilearn.image import clean_img
import pandas as pd

from nan_interp import nan_interp 

def confounds_control(subj_id, subjects_dir):
    input_path = f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    output_path = f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_filtered_space-MNI152NLin2009cAsym_desc-preproc_bold.nii'
    mask_path = f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    confounds_file = pd.read_csv(f'{subjects_dir}{subj_id}/func/{subj_id}_rest_confounds.csv').values

    confounds_file = nan_interp(confounds_file)

    filtered_img = clean_img(imgs=input_path,
                            mask_img=mask_path,
                            low_pass=0.1, 
                            high_pass=0.01, 
                            confounds=confounds_file,
                            detrend=False,
                            standardize=False,
                            t_r=2.0)

    filtered_img.to_filename(output_path)

def create_confounds_file(subj_id, subjects_dir):
    """
    input : fmriprep 전처리된 subjects 폴더 directory, 거기서 csv 파일 생성할 subject id(폴더명)
    return : X
    print : confounds_timeseries.tsv 파일이 존재하지 않을 경우 (subject_id, error)

    fmriprep 전처리 후 생성된 confounds_timeseries.tsv 파일에서
    필요한 confound만 가져와 새로운 csv 파일을 같은 directory에 생성 
    """
    error_subjects=[]
    try: 
        input_file = pd.read_csv(f'{subjects_dir}{subj_id}/func/{subj_id}_task-rest_desc-confounds_timeseries.tsv', sep = '\t')
        print(input_file)
    except FileNotFoundError as e:  error_subjects.append([{subj_id},e]) 
    print("error subjects : ", error_subjects)

#subjects_dir = 'C:/Users/janet/MyProjects/GNN_SCHZ/data'
subjects_dir = 'D:/MRI/UCLA_CNP/preprocess/fmriprep/derivatives/'
confounds_control('sub-10171', subjects_dir)
#create_confounds_file('sub-10171', subjects_dir)