from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
import nibabel as nib
import os
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.interfaces.fmriprep import load_confounds

from utils.viz_correlation_matrix import viz_correlation_matrix


# def create_FC(subj_dir, subj_id):
#     atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1, data_dir='data/atlas')
#     atlas.labels = np.insert(atlas.labels, 0, "Background")
#     fmriimg = nib.load(f'{subj_dir}/{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    
#     confounds, sample_mask = load_confounds(
#         fmriimg,
#         strategy=["high_pass", "motion", "wm_csf", "scrub", "global_signal"],
#         motion="basic",
#         wm_csf="basic",
#         global_signal="basic",
#         scrub=5,
#         fd_threshold=0.5,
#         std_dvars_threshold=1.5,
#     )
#     print("confounds : ",confounds.columns)
#     print(
#     f"After scrubbing, {sample_mask.shape[0]} "
#     f"out of {confounds.shape[0]} volumes remains"
# )

#     # atlas 반영된 mask를 fmriimg에 적용해 region별로 signal 추출 
#     masker = NiftiLabelsMasker(
#         labels_img=atlas.maps,
#         standardize="zscore_sample",
#         standardize_confounds="zscore_sample",
#         memory="nilearn_cache",
#         verbose=5,
#         low_pass=0.1
#     )
#     time_series = masker.fit_transform(fmriimg, 
#                                     confounds=confounds, sample_mask=sample_mask)



#     correlation_measure = ConnectivityMeasure(
#         kind="correlation",
#         standardize="zscore_sample",
#     )
#     FC = correlation_measure.fit_transform([time_series])[0]

#     viz_correlation_matrix(FC, atlas.labels)

fmriimg = nib.load(f'{subj_dir}/{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')

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