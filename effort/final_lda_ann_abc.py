#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

########################################### a jirak ################################################
# train data
data=pd.read_excel('C:/Users/CHOI/Desktop/train_1L_1000.xlsx') # b,c가 정상데이터로 인식 

X=data.drop(['target','type','m'],axis=1)
y=data.filter(['target'])
z=data.filter(['type'])

X_a=X.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])

row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()

# one-hot     
# one-hot     
for i in range(0,row-1):
    if z[i]==1: 
        yy[i]=yy[i]
    elif z[i]==2: 
        yy[i]=0
    else:         
        yy[i]=0

yy=pd.DataFrame(yy)

print('yy')
print(yy)


#test data
data_te=pd.read_excel('C:/Users/CHOI/Desktop/test_1L_200.xlsx')

X_te=data_te.drop(['target','type','m'],axis=1)

X_a_te=X_te.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])
y_te=data_te.filter(['target'])
z_te=data_te.filter(['type'])


row_te=y_te.shape[0]
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()

for i in range(0,row_te-1):
    if z_te[i]==1: #a
        yy_te[i]=yy_te[i]
    elif z_te[i]==2: #b
        yy_te[i]=0
    elif z_te[i]==3: #c
        yy_te[i]=0


yy_te=pd.DataFrame(yy_te)
print('yy_te')
print(yy_te)
###################################tn_lda_a#############################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda1 = lda.fit(X_a,y)
Xa_lda=lda1.transform(X_a)
print(Xa_lda.shape)
print(lda.explained_variance_ratio_)

Xa_lda_df = pd.DataFrame(Xa_lda)
X=pd.concat([Xa_lda_df],axis=1)
print(X.shape)


#################################LDA_te_a##################################

Xa_lda_te=lda1.transform(X_a_te)
print(Xa_lda_te.shape)
Xa_lda_te_df = pd.DataFrame(Xa_lda_te)
X_te=pd.concat([Xa_lda_te_df],axis=1)
print(X_te.shape)


#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

keras.utils.set_random_seed(10)

classifier = Sequential()

from keras.utils import to_categorical
yyy=to_categorical(yy)
yyy_te=to_categorical(yy_te)


#dense, optimizer, kernel_intializer

# Adding the input layer and first hidden layer / Dense 노드의 수 
# layer 값이 너무 많아도 gradient vanish를 유발할 수 있어서 조심 / input_dim 9개의 input /
classifier.add(Dense(64, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
# Adding the second hidden layer
classifier.add(Dense(2, kernel_initializer='he_normal', activation = 'leaky_relu'))
# Adding the output layer Dense 최종 아웃풋 층을 지정  18개(고장 위치 a,b,c) 값이 나와야함
classifier.add(Dense(12, kernel_initializer='he_normal', activation = 'softmax'))
classifier.summary()
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set

# yy 클래스이름 
results=classifier.fit(X, yyy, batch_size = 10, epochs = 200, validation_split = 0.2)


for i in range(500):

    if results.history['val_accuracy'][i] > results.history['val_accuracy'][i-1]:
        from keras.models import load_model
        classifier.save('C:/Users/CHOI/Desktop/Python/jirak_a.h5')
    else:
        pass

print(max(results.history['accuracy']))
print(max(results.history['val_accuracy']))
plt.plot(results.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()  


####
loss_and_metrics = classifier.evaluate(X_te, yyy_te, batch_size=1)
where = classifier.predict(X_te,batch_size=1)

print('')
print('loss_and_metrics : ' + str(loss_and_metrics))
print(str(where))
#####

#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


########################################### b jirak ################################################
# train data
data=pd.read_excel('C:/Users/CHOI/Desktop/train_1L_1000.xlsx') # b,c가 정상데이터로 인식 

X=data.drop(['target','type','m'],axis=1)
y=data.filter(['target'])
z=data.filter(['type'])

X_b=X.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])

row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()

# one-hot      
for i in range(0,row-1):
    if z[i]==1: 
        yy[i]=0
    elif z[i]==2: 
        yy[i]=yy[i]
    else:         
        yy[i]=0

yy=pd.DataFrame(yy)

print('yy')
print(yy)


#test data
data_te=pd.read_excel('C:/Users/CHOI/Desktop/test_1L_200.xlsx')

X_te=data_te.drop(['target','type','m'],axis=1)
X_b_te=X_te.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
y_te=data_te.filter(['target'])
z_te=data_te.filter(['type'])


row_te=y_te.shape[0]
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()

for i in range(0,row_te-1):
    if z_te[i]==1: #a
        yy_te[i]=0
    elif z_te[i]==2: #b
        yy_te[i]=yy_te[i]
    elif z_te[i]==3: #c
        yy_te[i]=0


yy_te=pd.DataFrame(yy_te)
print('yy_te')
print(yy_te)

###################################tn_lda_b#############################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda2 = lda.fit(X_b,y)
Xb_lda=lda2.transform(X_b)
print(Xb_lda.shape)
print(lda.explained_variance_ratio_)

Xb_lda_df = pd.DataFrame(Xb_lda)

X=pd.concat([Xb_lda_df],axis=1)
print(X.shape)


#################################LDA_te_b##################################

Xb_lda_te=lda2.transform(X_b_te)
print(Xb_lda_te.shape)
Xb_lda_te_df = pd.DataFrame(Xb_lda_te)
X_te=pd.concat([Xb_lda_te_df],axis=1)
print(X_te.shape)


#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

keras.utils.set_random_seed(10)

classifier = Sequential()

from keras.utils import to_categorical
yyy=to_categorical(yy)
yyy_te=to_categorical(yy_te)


#dense, optimizer, kernel_intializer

# Adding the input layer and first hidden layer / Dense 노드의 수 
# layer 값이 너무 많아도 gradient vanish를 유발할 수 있어서 조심 / input_dim 9개의 input /
classifier.add(Dense(16, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
# Adding the second hidden layer
classifier.add(Dense(4, kernel_initializer='he_normal', activation = 'leaky_relu'))
# Adding the output layer Dense 최종 아웃풋 층을 지정  18개(고장 위치 a,b,c) 값이 나와야함
classifier.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
classifier.summary()
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set

# yy 클래스이름 
results=classifier.fit(X, yyy, batch_size = 10, epochs = 150, validation_split = 0.2)


for i in range(150):

    if results.history['val_accuracy'][i] > results.history['val_accuracy'][i-1]:
        from keras.models import load_model
        classifier.save('C:/Users/CHOI/Desktop/Python/jirak_b.h5')
    else:
        pass

print(max(results.history['accuracy']))
print(max(results.history['val_accuracy']))
plt.plot(results.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()  


####
loss_and_metrics = classifier.evaluate(X_te, yyy_te, batch_size=1)
where = classifier.predict(X_te,batch_size=1)

print('')
print('loss_and_metrics : ' + str(loss_and_metrics))
print(str(where))
#####





########################################### c jirak ################################################
# train data
data=pd.read_excel('C:/Users/CHOI/Desktop/train_1L_1000.xlsx') # b,c가 정상데이터로 인식 

X=data.drop(['target','type','m'],axis=1)
y=data.filter(['target'])
z=data.filter(['type'])

X_c=X.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])

row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()

# one-hot     
for i in range(0,row-1):
    if z[i]==1: 
        yy[i]=0
    elif z[i]==2: 
        yy[i]=0
    else:         
        yy[i]=yy[i]

yy=pd.DataFrame(yy)

print('yy')
print(yy)


#test data
data_te=pd.read_excel('C:/Users/CHOI/Desktop/test_1L_200.xlsx')

X_te=data_te.drop(['target','type','m'],axis=1)
X_c_te=X_te.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])
y_te=data_te.filter(['target'])
z_te=data_te.filter(['type'])


row_te=y_te.shape[0]
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()

for i in range(0,row_te-1):
    if z_te[i]==1: #a
        yy_te[i]=0
    elif z_te[i]==2: #b
        yy_te[i]=0
    elif z_te[i]==3: #c
        yy_te[i]=yy_te[i]


yy_te=pd.DataFrame(yy_te)
print('yy_te')
print(yy_te)
###################################tn_lda_c#############################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda3 = lda.fit(X_c,y)
Xc_lda=lda3.transform(X_c)
print(Xc_lda.shape)
print(lda.explained_variance_ratio_)

Xc_lda_df = pd.DataFrame(Xc_lda)
X=pd.concat([Xc_lda_df],axis=1)
print(X.shape)


#################################LDA_te_c##################################
Xc_lda_te=lda3.transform(X_c_te)
print(Xc_lda_te.shape)

Xc_lda_te_df = pd.DataFrame(Xc_lda_te)
X_te=pd.concat([Xc_lda_te_df],axis=1)
print(X_te.shape)


#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

keras.utils.set_random_seed(10)

classifier = Sequential()

from keras.utils import to_categorical
yyy=to_categorical(yy)
yyy_te=to_categorical(yy_te)


#dense, optimizer, kernel_intializer

# Adding the input layer and first hidden layer / Dense 노드의 수 
# layer 값이 너무 많아도 gradient vanish를 유발할 수 있어서 조심 / input_dim 9개의 input /
classifier.add(Dense(4, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
# Adding the second hidden layer
classifier.add(Dense(8, kernel_initializer='he_normal', activation = 'leaky_relu'))
# Adding the output layer Dense 최종 아웃풋 층을 지정  18개(고장 위치 a,b,c) 값이 나와야함
classifier.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
classifier.summary()
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set

# yy 클래스이름 
results=classifier.fit(X, yyy, batch_size = 10, epochs = 150, validation_split = 0.2)


for i in range(150):

    if results.history['val_accuracy'][i] > results.history['val_accuracy'][i-1]:
        from keras.models import load_model
        classifier.save('C:/Users/CHOI/Desktop/Python/jirak_c.h5')
    else:
        pass

print(max(results.history['accuracy']))
print(max(results.history['val_accuracy']))
plt.plot(results.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()  


####
loss_and_metrics = classifier.evaluate(X_te, yyy_te, batch_size=1)
where = classifier.predict(X_te,batch_size=1)

print('')
print('loss_and_metrics : ' + str(loss_and_metrics))
print(str(where))
#####
