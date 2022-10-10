import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import joblib

def train_gnd(path, epoch):
    print("ground file start!")

    data_g = pd.read_excel(path + '/dataset/train 1L 1000.xlsx')

    X_g = data_g.drop(['target','type','m'],axis=1)
    y_g = data_g.filter(['target'])
    z_g = data_g.filter(['type'])

    row_g = y_g.shape[0]
    yy_g = y_g.to_numpy()
    z_g = z_g.to_numpy()

    lda = LinearDiscriminantAnalysis(n_components=3)

    keras.utils.set_random_seed(10)
    modela = Sequential()
    modelb = Sequential()
    modelc = Sequential()
    
    ########################################### a 지락 ###########################################
    X_a = X_g.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am',
                      'VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])

    yy_a = np.zeros((row_g, 1))

    for i in range(0, row_g):
        if z_g[i] == 1: 
            yy_a[i] = yy_g[i]
        elif z_g[i] == 2: 
            yy_a[i] = 0
        elif z_g[i] == 3:
            yy_a[i] = 0
        else:         
            yy_a[i] = 0

    yy_a = pd.DataFrame(yy_a)

    lda1 = lda.fit(X_a,yy_a)
    Xa_lda=lda1.transform(X_a)
    Xa = pd.DataFrame(Xa_lda)

    joblib.dump(lda1,path + '/model/lda1.pkl')
    
    yyy_a = to_categorical(yy_a)

    modela.add(Dense(16, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
    modela.add(Dense(256, kernel_initializer='he_normal', activation = 'leaky_relu'))
    modela.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))

    modela.summary()

    modela.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    results=modela.fit(Xa, yyy_a, batch_size = 10, epochs = epoch, validation_split = 0.2)

    hist_val = 0

    for i in range(epoch):
        if results.history['val_accuracy'][i] > hist_val:
            hist_val = results.history['val_accuracy'][i]
            modela.save(path + '/model/ground_a.h5')
        else:
            pass

    ########################################### b 지락 ###########################################
    X_b = X_g.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm',
                      'VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])

    yy_b = np.zeros((row_g, 1))

    for i in range(0, row_g):
        if z_g[i] == 1: 
            yy_b[i] = 0
        elif z_g[i] == 2: 
            yy_b[i] = yy_g[i]
        elif z_g[i] == 3:
            yy_b[i] = 0
        else:         
            yy_b[i] = 0

    yy_b = pd.DataFrame(yy_b)

    lda2 = lda.fit(X_b,yy_b)
    Xb_lda = lda2.transform(X_b)
    Xb = pd.DataFrame(Xb_lda)

    joblib.dump(lda2,path + '/model/lda2.pkl')

    yyy_b = to_categorical(yy_b)

    modelb.add(Dense(64, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
    modelb.add(Dense(16, kernel_initializer='he_normal', activation = 'leaky_relu'))
    modelb.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
    modelb.summary()

    modelb.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    results=modelb.fit(Xb, yyy_b, batch_size = 10, epochs = epoch, validation_split = 0.2)

    hist_val = 0

    for i in range(epoch):
        if results.history['val_accuracy'][i] > hist_val:
            hist_val = results.history['val_accuracy'][i]
            modelb.save(path + '/model/ground_b.h5')
        else:
            pass

    ########################################### c 지락 ###########################################
    X_c = X_g.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm',
                      'VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])

    yy_c = np.zeros((row_g, 1))

    for i in range(0,row_g):
        if z_g[i] == 1: 
            yy_c[i] = 0
        elif z_g[i] == 2: 
            yy_c[i] = 0
        elif z_g[i] == 3:
            yy_c[i] = yy_g[i]
        else:
            yy_c[i] = 0

    yy_c = pd.DataFrame(yy_c)

    lda3 = lda.fit(X_c,yy_c)
    Xc_lda=lda3.transform(X_c)
    Xc = pd.DataFrame(Xc_lda)

    joblib.dump(lda3,path + '/model/lda3.pkl')
    
    yyy_c = to_categorical(yy_c)

    modelc.add(Dense(4, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
    modelc.add(Dense(32, kernel_initializer='he_normal', activation = 'leaky_relu'))
    modelc.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
    modelc.summary()

    modelc.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    results=modelc.fit(Xc, yyy_c, batch_size = 10, epochs = epoch, validation_split = 0.2)

    hist_val = 0

    for i in range(epoch):
        if results.history['val_accuracy'][i] > hist_val:
            hist_val = results.history['val_accuracy'][i]
            modelc.save(path + '/model/ground_c.h5')
        else:
            pass
        
    print("ground file complete!")
    
if __name__ == "__main__":
    train_gnd(10)