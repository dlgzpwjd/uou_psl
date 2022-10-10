# k-최근접 이웃 실습 (fault_feature)
import os

currentpath = os.getcwd()
print(currentpath)


# 데이터 불러오기
import pandas as pd
file1 = pd.read_excel('C:/Users/gagri/Desktop/PYTHON/excel/feature_normal.xls')
file2 = pd.read_excel('C:/Users/gagri/Desktop/PYTHON/excel/fault_feature_jirak.xls')
file3 = pd.read_excel('C:/Users/gagri/Desktop/PYTHON/excel/fault_feature_seongan.xls')

df = pd.concat([file1, file2, file3])

print(df)


target = []

for idx, row in df.iterrows():
    if row['type'] == 0.0:
        target.append(0)
    elif row['type'] == 1.0 or row['type'] == 2.0 or row['type'] == 3.0:
        target.append(1)
    else:
        target.append(2)

df['type'] = target

print(df)


# 피처 / 타깃 데이터 지정
y = df.filter(['type']) 
X = df.drop(['target','type','m'], axis=1)



# 트레이닝 / 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=0)

# 데이터 학습 (k-최근접이웃)
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=4)
clf_knn.fit(X_tn, y_tn)

# 데이터 예측
knn_pred = clf_knn.predict(X_te)
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



#grid
estimator_KNN = KNeighborsClassifier(algorithm='auto')
knn_grid_set_up = {'n_neigbors': (1,10,1),
                   'classifier_leaf_size': (20,40,1), 'p': (1,2),
                   'classifier_weights': ('uniform', 'distance')
                   }

grid_search_KNN = GridSearchCV(
    estimator=estimator_KNN,
    param_grid=knn_grid_set_up,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
)
knn_grid.fit(x_train, y_train)
