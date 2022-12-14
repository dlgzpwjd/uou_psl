import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import joblib

def train_line(path, epoch):
    print("line file start!")

    data=pd.read_excel(path + '/dataset/train 2L 1000.xls')

    X_l = data.drop(['target','type'],axis=1)
    y_l = data.filter(['target'])
    z_l = data.filter(['type'])

    row_l = y_l.shape[0]
    yy_l = y_l.to_numpy()
    z_l = z_l.to_numpy()

    lda = LinearDiscriminantAnalysis(n_components=3)
    keras.utils.set_random_seed(10)
    modela = Sequential()
    modelb = Sequential()
    modelc = Sequential()

    ########################################### ab 선간 ###########################################
    X_ab=X_l.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am',
                     'VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph','VA_bm','VA_bph',
                     'IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm',
                     'IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])

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
    Xab = pd.DataFrame(Xab_lda)

    joblib.dump(lda4,path + '/model/lda4.pkl')
    
    yyy_ab=to_categorical(yy_ab)

    modela.add(Dense(512, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
    modela.add(Dense(64, kernel_initializer='he_normal', activation = 'leaky_relu'))
    modela.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
    modela.summary()
    modela.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    results=modela.fit(Xab, yyy_ab, batch_size = 10, epochs = epoch, validation_split = 0.2)

    hist_val = 0

    for i in range(epoch):
        if results.history['val_accuracy'][i] > hist_val:
            hist_val = results.history['val_accuracy'][i]
            modela.save(path + '/model/line_ab.h5')
        else:
            pass

    ########################################### bc 선간 ###########################################
    X_bc = X_l.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VA_cm',
                       'VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_bm','VC_bph',
                       'IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph','VC_cm','VC_cph','IC_cm',
                       'IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])

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
    Xbc = pd.DataFrame(Xbc_lda)

    joblib.dump(lda5,path + '/model/lda5.pkl')
    
    yyy_bc = to_categorical(yy_bc)

    modelb.add(Dense(4, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
    modelb.add(Dense(256, kernel_initializer='he_normal', activation = 'leaky_relu'))
    modelb.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
    modelb.summary()
    modelb.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    results=modelb.fit(Xbc, yyy_bc, batch_size = 10, epochs = epoch, validation_split = 0.2)

    hist_val = 0

    for i in range(epoch):
        if results.history['val_accuracy'][i] > hist_val:
            hist_val = results.history['val_accuracy'][i]
            modelb.save(path + '/model/line_bc.h5')
        else:
            pass

    ########################################### ca 선간 ###########################################
    X_ca=X_l.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm',
                     'VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph','VA_am','VA_aph',
                     'IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am',
                     'IC_aph','VD_am','VD_aph','ID_am','ID_aph'])

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
    Xca = pd.DataFrame(Xca_lda)

    joblib.dump(lda6,path + '/model/lda6.pkl')
    
    yyy_ca = to_categorical(yy_ca)

    modelc.add(Dense(512, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
    modelc.add(Dense(256, kernel_initializer='he_normal', activation = 'leaky_relu'))
    modelc.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
    modelc.summary()
    modelc.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    results=modelc.fit(Xca, yyy_ca, batch_size = 10, epochs = epoch, validation_split = 0.2)

    hist_val = 0

    for i in range(epoch):
        if results.history['val_accuracy'][i] > hist_val:
            hist_val = results.history['val_accuracy'][i]
            modelc.save(path + '/model/line_ca.h5')
        else:
            pass
        
    print("line file complete!")
    
if __name__ == "__main__":
    train_line(10)