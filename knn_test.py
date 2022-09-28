import pandas as pd
import numpy as np

data_g=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_jirak.xls')
data_l=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_seongan.xls')
data_n=pd.read_excel('C:/Users/user/Desktop/psl/feature_normal.xls')

df = pd.concat([data_g, data_l, data_n])

print(df)

target = []

for idx, row in df.iterrows():
    if row['type'] == 0:
        target.append(0)
    elif row['type'] == 1 or row['type'] == 2 or row['type'] == 3:
        target.append(1)
    else:
        target.append(2)

df['type'] = target

print(df)


# 피처 / 타깃 데이터 지정
y = df.filter(['type']) 
X = df.drop(['target','type','m'], axis=1)


import joblib
knn_test=joblib.load('./knn_model.pkl')
knn_pred = knn_test.predict(X)
print(knn_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, knn_pred)
print(accuracy)