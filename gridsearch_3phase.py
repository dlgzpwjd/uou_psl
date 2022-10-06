import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import to_categorical

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature 2L_1000.xls')

epochs = 200

X_l = data.drop(['target','type'],axis=1)
y_l = data.filter(['target'])  # to_categorical -> target 1~12 까지 되어있으면 칸을 13칸 줘 0~12 --> 그래서 -1을 해줘서 범위를 0~11로 변환
z_l = data.filter(['type'])

row_l = y_l.shape[0]
yy_l = y_l.to_numpy()
z_l = z_l.to_numpy()

lda = LinearDiscriminantAnalysis(n_components=3)

X_ca=X_l.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph','VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])

yy_ca = np.zeros((row_l, 1))

for i in range(0,row_l):
    if z_l[i]==4: 
        yy_ca[i]=0
    elif z_l[i]==5:
        yy_ca[i]=0
    elif z_l[i]==6:
        yy_ca[i]=yy_l[i]
    else:
        yy_ca[i]=0

yy_ca=pd.DataFrame(yy_ca)

lda6 = lda.fit(X_ca,yy_ca)
Xca_lda = lda6.transform(X_ca)
Xca_lda_df = pd.DataFrame(Xca_lda)
Xca = pd.concat([Xca_lda_df],axis=1)

yyy_ca = to_categorical(yy_ca)
keras.utils.set_random_seed(10)

# SJ 서치 시작 준비~~~~
dense_var1 = [2, 4, 8, 16, 32, 64, 128, 256, 512]
dense_var2 = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# 빈 데이터 프레임 생성
columns = ['dense1', 'dense2', 'accuracy', 'val_accuracy']
conclusion = pd.DataFrame(columns=columns)

# SJ 서치 (for 문 이용), Sequential : 순차적으로 레이어 층을 더해주는 모델
for dense1 in dense_var1:
    for dense2 in dense_var2:
        classifier = Sequential()


# ANN (Artificial Neural Network)

        classifier.add(Dense(dense1, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
        classifier.add(Dense(dense2, kernel_initializer='he_normal', activation = 'leaky_relu'))
        classifier.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        results=classifier.fit(Xca, yyy_ca, batch_size = 10, epochs = epochs, validation_split = 0.2)
        
        # SJ 서치 어디까지 됐는지 편하게 확인
        print('dense1 : %.1d / dense2 : %.1d Complete!!' %(dense1, dense2))

        # 엑셀에 내용을 저장하기 위한 준비
        conclusion_2 = pd.DataFrame({'dense1': [dense1],
                                    'dense2': [dense2],
                                    'accuracy': [max(results.history['accuracy'])],
                                    'val_accuracy':[max(results.history['val_accuracy'])]})


        # 결과를 모두 합침
        conclusion = pd.concat([conclusion, conclusion_2], ignore_index=True)


        # 마무리 : 엑셀에 저장
conclusion.to_excel('conclusion_ca.xlsx')

# conclusion 이용해서 제일 좋은 성능 나오는 dense1 dense2 를 빼오는 코딩 1줄


X_bc = X_l.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])

yy_bc = np.zeros((row_l, 1))

for i in range(0,row_l):
    if z_l[i] == 4: 
        yy_bc[i] = 0
    elif z_l[i] == 5:
        yy_bc[i] = yy_l[i]
    elif z_l[i] == 6:
        yy_bc[i] = 0
    else:
        yy_bc[i] = 0

yy_bc = pd.DataFrame(yy_bc)

lda5 = lda.fit(X_bc,yy_bc)
Xbc_lda = lda5.transform(X_bc)
Xbc_lda_df = pd.DataFrame(Xbc_lda)
Xbc = pd.concat([Xbc_lda_df],axis=1)

yyy_bc = to_categorical(yy_bc)

# 빈 데이터 프레임 생성
columns = ['dense1', 'dense2', 'accuracy', 'val_accuracy']
conclusion = pd.DataFrame(columns=columns)

# SJ 서치 (for 문 이용), Sequential : 순차적으로 레이어 층을 더해주는 모델
for dense1 in dense_var1:
    for dense2 in dense_var2:
        classifier2 = Sequential()

# ANN (Artificial Neural Network)

        classifier2.add(Dense(dense1, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
        classifier2.add(Dense(dense2, kernel_initializer='he_normal', activation = 'leaky_relu'))
        classifier2.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
        classifier2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        results2=classifier2.fit(Xbc, yyy_bc, batch_size = 10, epochs = epochs, validation_split = 0.2)
        
        # SJ 서치 어디까지 됐는지 편하게 확인
        print('dense1 : %.1d / dense2 : %.1d Complete!!' %(dense1, dense2))

        # 엑셀에 내용을 저장하기 위한 준비
        conclusion_2 = pd.DataFrame({'dense1': [dense1],
                                    'dense2': [dense2],
                                    'accuracy': [max(results2.history['accuracy'])],
                                    'val_accuracy':[max(results2.history['val_accuracy'])]})

        # 결과를 모두 합침
        conclusion = pd.concat([conclusion, conclusion_2], ignore_index=True)

        # 마무리 : 엑셀에 저장
conclusion.to_excel('conclusion_bc.xlsx')

X_ab=X_l.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])

yy_ab = np.zeros((row_l, 1))

for i in range(0,row_l):
    if z_l[i] == 4:
        yy_ab[i] = yy_l[i]
    elif z_l[i]==5:
        yy_ab[i] = 0
    elif z_l[i] == 6:
        yy_ab[i] = 0
    else:
        yy_ab[i] = 0

yy_ab = pd.DataFrame(yy_ab)

lda4 = lda.fit(X_ab,yy_ab)
Xab_lda = lda4.transform(X_ab)
Xab_lda_df = pd.DataFrame(Xab_lda)
Xab = pd.concat([Xab_lda_df],axis=1)

yyy_ab=to_categorical(yy_ab)

# 빈 데이터 프레임 생성
columns = ['dense1', 'dense2', 'accuracy', 'val_accuracy']
conclusion = pd.DataFrame(columns=columns)

# SJ 서치 (for 문 이용), Sequential : 순차적으로 레이어 층을 더해주는 모델
for dense1 in dense_var1:
    for dense2 in dense_var2:
        classifier3 = Sequential()

# ANN (Artificial Neural Network)

        classifier3.add(Dense(dense1, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
        classifier3.add(Dense(dense2, kernel_initializer='he_normal', activation = 'leaky_relu'))
        classifier3.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
        classifier3.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        results3=classifier3.fit(Xab, yyy_ab, batch_size = 10, epochs = epochs, validation_split = 0.2)
        
        # SJ 서치 어디까지 됐는지 편하게 확인
        print('dense1 : %.1d / dense2 : %.1d Complete!!' %(dense1, dense2))

        # 엑셀에 내용을 저장하기 위한 준비
        conclusion_2 = pd.DataFrame({'dense1': [dense1],
                                    'dense2': [dense2],
                                    'accuracy': [max(results3.history['accuracy'])],
                                    'val_accuracy':[max(results3.history['val_accuracy'])]})

        # 결과를 모두 합침
        conclusion = pd.concat([conclusion, conclusion_2], ignore_index=True)

        # 마무리 : 엑셀에 저장
conclusion.to_excel('conclusion_ab.xlsx')