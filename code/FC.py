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


def create_FC(subj_dir, subj_id,
              conntype='correlation', space='data', 
              savets=False, nomat=False, dtr=False, stdz=False):
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1, data_dir='data/atlas')
    atlas.labels = np.insert(atlas.labels, 0, "Background")
    atlasimg = atlas.maps
    fmriimg = nib.load(f'{subj_dir}/{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    maskimg = nib.load(f'{subj_dir}/{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    
    confounds, sample_mask = load_confounds(
        fmriimg,
        strategy=["high_pass", "motion", "wm_csf", "scrub", "global_signal"],
        motion="basic",
        wm_csf="basic",
        global_signal="basic",
        scrub=5,
        fd_threshold=0.5,
        std_dvars_threshold=1.5,
    )
    print("confounds : ",confounds.columns)
    print(
    f"After scrubbing, {sample_mask.shape[0]} "
    f"out of {confounds.shape[0]} volumes remains"
)

    # atlas 반영된 mask를 fmriimg에 적용해 region별로 signal 추출 
    masker = NiftiLabelsMasker(
        labels_img=atlasimg,
        mask_img=maskimg,

        smoothing_fwhm=None,
        standardize=stdz, #False
        standardize_confounds="zscore_sample", #True
        detrend=dtr,
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.0,
        
        resampling_target=space,
        memory="nilearn_cache",
        background_label=0,
        verbose=1
    )
    time_series = masker.fit_transform(fmriimg, 
                                    confounds=confounds, sample_mask=sample_mask)

    # mask the labimg so that there are no regions that dont have data 
    if space == 'data':
        # resample_to_image(source, target)
        # assume here that the mask is also fmri space
        resamplabs = resample_to_img(atlasimg,maskimg,interpolation='nearest')
        resampmask = resample_to_img(maskimg,maskimg,interpolation='nearest')
    else:
        resamplabs = resample_to_img(atlasimg,atlasimg,interpolation='nearest')
        resampmask = resample_to_img(maskimg,atlasimg,interpolation='nearest')
        
    # mask
    resamplabsmasked = apply_mask(resamplabs,resampmask)

    correlation_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )
    FC = correlation_measure.fit_transform([time_series])[0]

    viz_correlation_matrix(FC, atlas.labels)
    return FC

def makeFC(subj_dir, dataset_name):
    subj_list = sorted(os.listdir(subj_dir))

    error_subjects=[]
    for subj_id in subj_list:
        try :
            print(f'==============={subj_id}=======================')
            FC = create_FC(subj_dir, subj_id, conntype='correlation', space='data', savets=False, nomat=False, dtr=False, stdz=False)
            np.save(f'data/tempFC/{dataset_name}/FC100_{subj_id}.npy',FC)
        except ValueError as e :
            error_subjects.append([{subj_id},e])
        except FileNotFoundError as e:
            error_subjects.append([{subj_id},e])
    
    print(error_subjects)


dataset_name = 'UCLA_CNP'
subj_dir = f'/home/jihoo/{dataset_name}/COBRE/derivatives'  # 전처리된 fmri image가 저장된 경로
makeFC(subj_dir, dataset_name)