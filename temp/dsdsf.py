from sklearn.metrics import balanced_accuracy_score, recall_score , confusion_matrix
import numpy as np

y_true=np.array([0,0,0])
y_pred=np.array([0,0,0])

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()