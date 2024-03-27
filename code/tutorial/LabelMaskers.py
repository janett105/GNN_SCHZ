import matplotlib.pyplot as plt
from nilearn import datasets
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib

dataset_name = 'UCLA_CNP'
subj_dir = f'D:/MRI/{dataset_name}/preprocess/fmriprep/derivatives'
subj_id='sub-10159'

maskimg = nib.load(f'{subj_dir}/{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
fmriimg = f'{subj_dir}/{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1, data_dir='data/atlas')
atlas.labels = np.insert(atlas.labels, 0, "Background")
atlasimg = atlas.maps

# Instantiate the masker with label image and label values
masker = NiftiLabelsMasker(
    atlasimg,
    mask_img=maskimg,
    labels=atlas.labels,

    smoothing_fwhm=4.0,
    standardize="zscore_sample", #False
    standardize_confounds=True, #True
    detrend=True, #None
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0,

    background_label=0,
    resampling_target='data'
)
# Visualize the atlas
# Note that we need to call fit prior to generating the mask
masker.fit(fmriimg)

# At this point, no functional image has been provided to the masker.
# We can still generate a report which can be displayed in a Jupyter
# Notebook, opened in a browser using the .open_in_browser() method,
# or saved to a file using the .save_as_html(output_filepath) method.
report = masker.generate_report()
report.open_in_browser()

signals = masker.transform(fmriimg)

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
for label_idx in range(100):
    ax.plot(
        signals[:, label_idx], linewidth=2, label=atlas.labels[label_idx + 1]
    )  # 0 is background
ax.legend(loc=2)
ax.set_title("Signals for first 3 regions")
plt.show()