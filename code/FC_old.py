import nilearn
import nibabel as nib

import os
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import resample_to_img
from nilearn import input_data, connectome

import pandas as pd
import numpy as np

# matrix 추출
def extract_mat(rsimg, maskimg, labelimg, conntype='correlation', space='labels', 
                savets=False, nomat=False, dtr=False, stdz=False):

    masker = NiftiLabelsMasker( labels_img=labelimg,
                                background_label=0,
                                smoothing_fwhm=None,
                                standardize=stdz, 
                                detrend=dtr,
                                mask_img=maskimg,
                                resampling_target=space,
                                verbose=1)

    # mask the labimg so that there are no regions that dont have data
    from nilearn.image import resample_to_img
    if space == 'data':
        # resample_to_image(source, target)
        # assume here that the mask is also fmri space
        resamplabs = resample_to_img(labelimg,maskimg,interpolation='nearest')
        resampmask = resample_to_img(maskimg,maskimg,interpolation='nearest')
    else:
        resamplabs = resample_to_img(labelimg,labelimg,interpolation='nearest')
        resampmask = resample_to_img(maskimg,labelimg,interpolation='nearest')
        
    # mask
    from nilearn.masking import apply_mask
    resamplabsmasked = apply_mask(resamplabs,resampmask)

    # get the unique labels list, other than 0, which will be first
    #reginparc = np.unique(resamplabs.get_fdata())[1:].astype(np.int)
    reginparc = np.unique(resamplabsmasked)[1:].astype(int)
    reglabs = list(reginparc.astype(str))
    
    reginorigparc = np.unique(labelimg.get_fdata())[1:].astype(int)
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
    time_series = masker.fit_transform(rsimg)
    if nomat:
        connmat = None
        conndf = None
    else:
        connobj = ConnectivityMeasure(kind=conntype)
        connmat = connobj.fit_transform([time_series])[0]
        conndf = get_con_df(connmat, reglabs)

    # if not saving time series, don't pass anything substantial, save mem
    if not savets:
        time_series = 42

    return conndf, connmat, time_series, reginparc

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

    np.fill_diagonal(raw_mat, 0)
    con_df = pd.DataFrame(raw_mat, index=roi_names, columns=roi_names)
    return con_df

def makeFC(rawdata_dir):
    atlas = 'AAL3v1_164parcels_1mm'

    # labelimg -> parcellation atlas 선택
    labelimg = nib.load('nilearn_data/AAL/AAL3v1_1mm.nii.gz')

    # subjec number for rsimg & ma
    subj_list = sorted(os.listdir(rawdata_dir))
    error_subjects=[]
    for subjnum in subj_list:
        try :
            print(f'==============={subjnum}=======================')
            # rsimg -> (resting state) fmri image 경로 입력
            rsimg = nib.load(f'{rawdata_dir}/{subjnum}/func/{subjnum}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')

            # maskimg -> rsimg에 해당하는 brain mask 경로 입력
            maskimg = nib.load(f'{rawdata_dir}/{subjnum}/func/{subjnum}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')  
        
            # connectivity matrix calculation
            conndf, connmat, time_series, reginparc = extract_mat(rsimg, maskimg, labelimg, conntype='correlation', space='data', savets=False, nomat=False, dtr=False, stdz=False)
               
            # connmat -> functional connectivity matrix 생성된 결과
            np.save(f'results/{atlas}/FC164_{subjnum}.npy',conndf)
        except ValueError as e :
            error_subjects.append([{subjnum},e])
        except FileNotFoundError as e:
            error_subjects.append([{subjnum},e])
    return error_subjects

# 전처리된 fmri image가 저장된 경로
rawdata_dir = '/home/jihoo/UCLA_CNP_FC/COBRE/derivatives'

error_subjects = makeFC(rawdata_dir)

print(error_subjects)