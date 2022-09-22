#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

#train data
data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature 2L_1000.xls')

X=data.drop(['target','type','m'], axis=1)
y=data.filter(['target'])
z=data.filter(['type'])

from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te, z_tn, z_te=train_test_split(X, y, z, test_size=0.2,shuffle=True, random_state=1)

X_a_tn=X_tn.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_am','VC_aph','IC_am','IC_aph','VC_am','VC_aph','IC_am','IC_aph','VC_bm','VC_bph','IC_bm','IC_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_am','VD_aph','ID_am','ID_aph','VD_am','VD_aph','ID_am','ID_aph','VD_bm','VD_bph','ID_bm','ID_bph','VD_bm','VD_bph','ID_bm','ID_bph'])

row=y_tn.shape[0]
yy_tn=y_tn.to_numpy()
z_tn=z_tn.to_numpy()
    
for i in range(0,row-1):
    if z_tn[i]==4:
        yy_tn[i]=yy_tn[i]
    elif z_tn[i]==5:
        yy_tn[i]=0
    elif z_tn[i]==6:
        yy_tn[i]=0
    else:
        yy_tn[i]=0
        
yy_tn=pd.DataFrame(yy_tn)
print(yy_tn.shape)

# test data
X_a_te=X_te.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_am','VC_aph','IC_am','IC_aph','VC_am','VC_aph','IC_am','IC_aph','VC_bm','VC_bph','IC_bm','IC_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_am','VD_aph','ID_am','ID_aph','VD_am','VD_aph','ID_am','ID_aph','VD_bm','VD_bph','ID_bm','ID_bph','VD_bm','VD_bph','ID_bm','ID_bph'])

row_te=y_te.shape[0]
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()
    
for i in range(0,row_te-1):
    if z_te[i]==4:
        yy_te[i]=yy_te[i]
    elif z_te[i]==5:
        yy_te[i]=0
    elif z_te[i]==6:
        yy_te[i]=0
    else:
        yy_te[i]=0
        
yy_te=pd.DataFrame(yy_te)
print(yy_te.shape)

#LDA_tn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components = 3)
lda4 = lda.fit(X_a_tn,yy_tn)

Xa_lda_tn=lda4.transform(X_a_tn)

print(Xa_lda_tn.shape)

print(lda.explained_variance_ratio_)

Xa_lda_tn_df = pd.DataFrame(Xa_lda_tn)

#LDA_te
Xa_lda_te=lda4.transform(X_a_te)

print(Xa_lda_te.shape)

Xa_lda_te_df = pd.DataFrame(Xa_lda_te)

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import to_categorical


keras.utils.set_random_seed(1)

model = Sequential()

    
yyy_tn=to_categorical(yy_tn)
yyy_te=to_categorical(yy_te)

# Adding the input layer and first hidden layer
model.add(Dense(64, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
# Adding the second hidden layer
model.add(Dense(32, kernel_initializer='he_normal', activation = 'leaky_relu'))
# Adding the second hidden layer
# model.add(Dense(32, kernel_initializer='he_normal', activation = 'leaky_relu'))
# Adding the output layer
model.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))

model.summary()

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set

epoch=150
results=model.fit(Xa_lda_tn, yyy_tn, batch_size = 10, epochs = epoch, validation_split = 0.2)

result=np.array(results.history['val_accuracy'])
acc=np.array(results.history['accuracy'])

for i in range(epoch):
    if result[i] > result[i-1]:
        model.save('C:/Python/uou_psl/hj_learning_ab.h5')
    else:
        pass
        
print('Max of acc: ',max(results.history['accuracy']))
print('Max of Val_acc: ',max(results.history['val_accuracy']))

plt.plot(results.history['val_accuracy'])
plt.plot(results.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val_acc', 'acc'], loc='lower right')
plt.savefig('test.png')
plt.show()

from keras.models import load_model
model = load_model("hj_learning_ab.h5")

model.evaluate(Xa_lda_te,yyy_te, batch_size = 1)
model.predict(Xa_lda_te,batch_size = 1)