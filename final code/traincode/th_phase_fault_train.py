import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import to_categorical

def train_3ph(epoch):

    print("3phase file start!")

    data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature 3L_300.xls') 

    X_abc=data.drop(['target','type'],axis=1)
    y=data.filter(['target']) - 1 
    z=data.filter(['type'])

    lda=LinearDiscriminantAnalysis(n_components=3)
    lda7 = lda.fit(X_abc,y)

    Xabc_lda=lda7.transform(X_abc)
    Xabc_lda_df = pd.DataFrame(Xabc_lda)
    Xabc=pd.concat([Xabc_lda_df],axis=1)

    yy=to_categorical(y)

    keras.utils.set_random_seed(10)
    model = Sequential()

    model.add(Dense(4, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
    model.add(Dense(32, kernel_initializer='he_normal', activation = 'leaky_relu'))
    model.add(Dense(12, kernel_initializer='he_normal', activation = 'softmax'))
    model.summary()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    #epoch = 200
    results=model.fit(Xabc, yy, batch_size = 10, epochs = epoch, validation_split = 0.2)

    hist_val = 0

    for i in range(epoch):
        if results.history['val_accuracy'][i] > hist_val:
            hist_val = results.history['val_accuracy'][i]
            model.save('./3phase.h5')
        else:
            pass
        
    print("3phase file complete!")
    
if __name__ == "__main__":
    train_3ph(10)