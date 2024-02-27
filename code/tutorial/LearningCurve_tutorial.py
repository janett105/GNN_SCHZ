import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits # dataset
from sklearn.svm import SVC # model

from sklearn.model_selection import LearningCurveDisplay, StratifiedKFold

def LearningCrv(estimators, X, y):
    """
    estimators : estimator lsit
    X : feature matrix (training/test split X)
    y : label (training/test split X)
    """
    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
        "score_type": "both", # test & train score plot
        "n_jobs": 4,

        "line_kw": {"marker": "o"}, 
        "std_display_style": "fill_between",
        "score_name": "Balanced Accuracy",
    }

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharey=True)
    for ax_idx, estimator in enumerate(estimators):
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax)
        handles, label = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ["Training Score", "Test Score"])
        ax.set_title(f"Learning Curve for {estimator.__class__.__name__}")

    plt.show()

#X, y = load_digits(return_X_y=True)
#svc = SVC(kernel="rbf", gamma=0.001)
#LearningCrv([svc], X, y)