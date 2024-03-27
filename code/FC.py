from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
import nibabel as nib
import os
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask

from utils.viz_correlation_matrix import viz_correlation_matrix


def create_FC(subj_dir, subj_id, t_r,
              dtr=True, stdz="zscore_sample", smth = 4.0, cnfstdz = True, confs=["motion","scrub", "wm_csf","global_signal"], connstdz = True,
              conntype='correlation', space='data', savets=False, nomat=False):
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1, data_dir='data/atlas')
    atlas.labels = np.insert(atlas.labels, 0, "Background")
    atlasimg = nib.load(atlas.maps)
    fmriimg = f'{subj_dir}/{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    maskimg = nib.load(f'{subj_dir}/{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
           
    # sample mask : 조건(scrubbing) 따라 제거된 volume 없는 경우 None, 있는 경우 제거되지 않은 volumes(time) index
    confounds, sample_mask = load_confounds(
        img_files=fmriimg,
        strategy=confs,
        motion="basic",
        wm_csf="basic",
        global_signal="basic",
        scrub=5,
        #fd_threshold=0.5,
        std_dvars_threshold=1.5,
    ) 

    if isinstance(sample_mask, np.ndarray):
        print(
            f"After scrubbing, {sample_mask.shape[0]} "
            f"out of {confounds.shape[0]} volumes remains")   
    else:     
        print(
            f"After scrubbing, {confounds.shape[0]} "
            f"out of {confounds.shape[0]} volumes remains")

    # atlas 반영된 mask를 fmriimg에 적용해 region별로 signal 추출 
    masker = NiftiLabelsMasker(
        labels_img=atlasimg,
        mask_img=maskimg,

        smoothing_fwhm=smth,
        standardize=stdz, #False
        standardize_confounds=cnfstdz, #True
        detrend=dtr, #None
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.0,
        
        resampling_target=space, #data : atlas가 data shape으로 resample
        memory="nilearn_cache",
        background_label=0,
        verbose=1
    )
    time_series = masker.fit_transform(fmriimg, 
                                    confounds=confounds, sample_mask=sample_mask)
    
    # print(f'resampling 전 atlas voxel size : {atlasimg.header.get_zooms()}')
    # print(f'resampling 전 mask voxel size : {maskimg.header.get_zooms()}')
    # mask the labimg so that there are no regions that dont have data 
    if space == 'data':
        # resample_to_image(source, target)
        # assume here that the mask is also fmri space
        resamplabs = resample_to_img(atlasimg,maskimg,interpolation='nearest')
        resampmask = resample_to_img(maskimg,maskimg,interpolation='nearest')
    else:
        resamplabs = resample_to_img(atlasimg,atlasimg,interpolation='nearest')
        resampmask = resample_to_img(maskimg,atlasimg,interpolation='nearest')   
    # print(f'resampling 후 atlas voxel size : {resamplabs.header.get_zooms()}')
    # print(f'resampling 후 mask voxel size : {resampmask.header.get_zooms()}')

    # mask
    resamplabsmasked = apply_mask(resamplabs,resampmask)
    
    print(f'resampling 전 atlas label 수 : {len(np.unique(atlasimg.get_fdata())[1:].astype(int))}') # reginorigparc
    print(f'resampling 후 atlas label 수 : {len(np.unique(resamplabs.get_fdata())[1:].astype(int))}')
    print(f'resampling 후 atlas에 mask 적용했을 때 label 수 : {len(np.unique(resamplabsmasked)[1:].astype(int))}') # reginparc, reglabs
    
    # get the unique labels list, other than 0, which will be first
    #reginparc = np.unique(resamplabs.get_fdata())[1:].astype(np.int)
    reginparc = np.unique(resamplabsmasked)[1:].astype(int)
    reglabs = list(reginparc.astype(str))

    reginorigparc = np.unique(atlasimg.get_fdata())[1:].astype(int)
    if len(reginparc) != len(reginorigparc):
        print('\n !!!WARNING!!! during resampling of label image, some of the'
              ' ROIs (likely very small) were interpolated out. Please take '
              'care to note which ROIs are present in the output data\n')
        print('ALTERNATIVELY, your parcellation is not in the same space'
              'as the bold data.\n')
        if abs(len(reginparc) - len(reginorigparc)) > 9:
            print('\nBASED ON QUICK HEURISTIC...I think your parcellation '
                  'is not in the right space. Please check that the two '
                  'images are aligned properly.')

    # Extract time series
    if nomat:
        FC_df = None
    else:
        correlation_measure = ConnectivityMeasure(kind=conntype,standardize=connstdz)
        FC = correlation_measure.fit_transform([time_series])[0]
        FC_df = get_con_df(FC, reglabs)

        new_labels = atlas.labels[reginorigparc]
        viz_correlation_matrix(FC, new_labels)
    
    return FC_df, time_series, reginparc

# connectivity dataframe
def get_con_df(raw_mat, roi_names):
    """
    takes a symmetrical connectivity matrix (e.g., numpy array) and a list of roi_names (strings)
    returns data frame with roi_names as index and column names
    e.g.
         r1   r2   r3   r4
    r1  0.0  0.3  0.7  0.2
    r2  0.3  0.0  0.6  0.5
    r3  0.7  0.6  0.0  0.9
    r4  0.2  0.5  0.9  0.0
    """
    # sanity check if matrix is symmetrical
    assert np.allclose(raw_mat, raw_mat.T, atol=1e-04, equal_nan=True), "matrix not symmetrical"

    np.fill_diagonal(raw_mat, 0) # 대각선 0으로 설정
    con_df = pd.DataFrame(raw_mat, index=roi_names, columns=roi_names)
    return con_df

def makeFC(subj_dir, dataset_name, t_r):
    subj_list = sorted(os.listdir(subj_dir))

    error_subjects=[]
    for subj_id in subj_list:
        if subj_id!='sub-10159': continue
        # try :
        print(f'==============={subj_id}=======================')
        FC_df, time_series, reginparc = create_FC (subj_dir, subj_id, t_r,
                        dtr=True, stdz="zscore_sample", smth = 4.0, cnfstdz = True, confs=["motion","scrub", "wm_csf","global_signal"], connstdz = True,
                        conntype='correlation', space='data', savets=False, nomat=False)
        np.save(f'D:/MRI/{dataset_name}/preprocess/fmriprep/FC/Schaefer2018_100parcels_7networks_1mm/FC100_{subj_id}.npy',FC_df.values)
        # except ValueError as e :
        #     error_subjects.append([{subj_id},e])
        # except FileNotFoundError as e:
        #     error_subjects.append([{subj_id},e])
    
    print(error_subjects)

dataset_name = 'UCLA_CNP'
subj_dir = f'D:/MRI/{dataset_name}/preprocess/fmriprep/derivatives'  # 전처리된 fmri image가 저장된 경로
t_r=2.0
makeFC(subj_dir, dataset_name, t_r) # Atlas : Schaefer2018_100parcels_7networks_1mm