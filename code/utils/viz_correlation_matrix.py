import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt

def viz_correlation_matrix(correlation_matrix, labels):
    np.fill_diagonal(correlation_matrix, 0)

    plotting.plot_matrix(
        correlation_matrix,
        figure=(10, 8),
        labels=labels, # background 제거한 labels여야 함
        vmax=0.8,
        vmin=-0.8,
        title="Functional Connectivity Matrix",
        reorder=True,
    )

    plt.show()