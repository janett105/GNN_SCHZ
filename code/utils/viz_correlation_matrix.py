import numpy as np
from nilearn import plotting

def viz_correlation_matrix(correlation_matrix, atlas_labels):
    np.fill_diagonal(correlation_matrix, 0)

    plotting.plot_matrix(
        correlation_matrix,
        figure=(10, 8),
        labels=atlas_labels[1:], # label[:,0]에 'Background'가 존재하므로 해당 부분 plot에서 제외
        vmax=0.8,
        vmin=-0.8,
        title="Functional Connectivity Matrix",
        reorder=True,
    )