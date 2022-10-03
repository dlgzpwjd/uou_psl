# k-최근접 이웃 실습 (fault_feature)

# 데이터 불러오기
import pandas as pd
data_g=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_jirak.xls')
data_l=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_seongan.xls')
data_n=pd.read_excel('C:/Users/user/Desktop/psl/feature_normal.xls')
data_p=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature 3L_100.xls')

df = pd.concat([data_g, data_l, data_n,data_p])

print(df)

target = []

for idx, row in df.iterrows():
    if row['type'] == 0:
        target.append(0)
    elif row['type'] == 1 or row['type'] == 2 or row['type'] == 3:
        target.append(1)
    elif row['type'] == 4 or row['type'] == 5 or row['type'] == 6:
        target.append(2)
    else:
        target.append(3)

df['type'] = target

print(df)


# 피처 / 타깃 데이터 지정
y = df.filter(['type']) 
X = df.drop(['target','type','m'], axis=1)

# 트레이닝 / 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,test_size=0.2,random_state=0)


# 데이터 학습 (k-최근접이웃)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_tn, y_tn)

import joblib
joblib.dump(knn,'./knn_model.pkl')

# 데이터 예측
knn_pred = knn.predict(X_te)
print(knn_pred)

# 정확도 평가
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_te, knn_pred)
print(accuracy)

# confusion matrix 확인
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_te, knn_pred)
print(conf_matrix)

# 분류 리포트 확인
from sklearn.metrics import classification_report
class_report = classification_report(y_te, knn_pred)
print(class_report)

