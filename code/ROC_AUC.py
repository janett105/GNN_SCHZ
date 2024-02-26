import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 유방암 데이터셋 로드
# data = datasets.load_breast_cancer()
# df = pd.DataFrame(data.data, columns = data.feature_names)
# df['target'] = data.target

# X = df.iloc[:, :-1] # target column을 제외한 모든 column을 feature로 사용
# y = df['target']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_score = model.predict(X_test) # 테스트 데이터셋에 대한 예측 값
# fpr, tpr, thresholds = roc_curve(y_test, y_score) # input 순서 : 실제 라벨, 예측 값
# print((fpr, tpr, thresholds))
# print(roc_auc_score(y_test, y_score))

# plt.plot(fpr, tpr)

# plt.xlabel('FP Rate')
# plt.ylabel('TP Rate')

# plt.show()

def ROC_threshold(val_y_true, val_y_score, test_y_true, test_y_score):
    val_fpr, val_tpr, val_thresholds = roc_curve(val_y_true, val_y_score)
    test_fpr, test_tpr, test_thresholds = roc_curve(test_y_true, test_y_score)
    
    # plt.plot(val_fpr, val_tpr)
    # plt.plot(test_fpr, test_tpr)

    # plt.xlabel('FP Rate')
    # plt.ylabel('TP Rate')

    # plt.show()

    return [val_thresholds[np.argmax(val_tpr - val_fpr)], test_thresholds[np.argmax(test_tpr - test_fpr)]]