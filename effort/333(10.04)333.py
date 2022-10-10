### 3상 단락 고장


#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel('C:/Users/Jendk/Desktop/asd/fault_feature 3L_300.xls') 

X=data.drop(['target','type'],axis=1)
X_abc=data.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph','VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])


#X_a=data.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph',] ['VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])
#X_b=data.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph',] ['VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
#X_c=data.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph',] ['VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])


y=data.filter(['target'])
z=data.filter(['type'])


row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()



# one-hot     
for i in range(0,row):  # 3상 단락 고장 
                   
    yy[i]=yy[i]

yy=pd.DataFrame(yy)
z_1=pd.DataFrame(z)

#test data
data_te=pd.read_excel('C:/Users/Jendk/Desktop/asd/fault_feature 3L_300.xls')

X_abc_te=data_te.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph','VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])


#X_a_te=data.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])
#X_b_te=data.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])
#X_c_te=data.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])


y_te=data_te.filter(['target'])
z_te=data_te.filter(['type'])

row_te=y_te.shape[0]
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()

for i in range(0,row_te-1): # 3상 단락 고장 
            
    yy_te[i]=yy_te[i]   


yy_te=pd.DataFrame(yy_te)


#tn_lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda.fit(X_abc,y)
#lda.fit(X_b,y)
#lda.fit(X_c,y)

Xabc_lda=lda.transform(X_abc)
#Xb_lda=lda.transform(X_b)
#Xc_lda=lda.transform(X_c)

Xabc_lda_df = pd.DataFrame(Xabc_lda)
#Xb_lda_df = pd.DataFrame(Xb_lda)
#Xc_lda_df = pd.DataFrame(Xc_lda)

X=pd.concat([Xabc_lda_df],axis=1)

Xabc_lda_te=lda.transform(X_abc_te)
#Xb_lda_te=lda.transform(X_b_te)
#Xc_lda_te=lda.transform(X_c_te)

Xabc_lda_te_df = pd.DataFrame(Xabc_lda_te)
#Xb_lda_te_df = pd.DataFrame(Xb_lda_te)
#Xc_lda_te_df = pd.DataFrame(Xc_lda_te)

X_te=pd.concat([Xabc_lda_te_df],axis=1)

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.utils import to_categorical
yyy=to_categorical(yy)
yyy_te=to_categorical(yy_te)

keras.utils.set_random_seed(10)

dense_var1 = [2, 4, 8, 16, 32, 64, 128, 256, 512]
dense_var2 = [2, 4, 8, 16, 32, 64, 128, 256, 512]

columns = ['dense1', 'dense2', 'accuracy', 'val_accuracy', 'test_accuracy']
conclusion = pd.DataFrame(columns=columns)

for dense1 in dense_var1:
    for dense2 in dense_var2:
        classifier = Sequential()

        #'leaky_relu' 'relu' 'elu'
        # https://yeomko.tistory.com/39 active funtion 

        # Adding the input layer and first hidden layer / Dense 노드의 수 
        # layer 값이 너무 많아도 gradient vanish를 유발할 수 있어서 조심 / input_dim 9개의 input /
        classifier.add(Dense(dense1, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
        # Adding the second hidden layer
        classifier.add(Dense(dense2, kernel_initializer='he_normal', activation = 'leaky_relu'))
        # Adding the third hidden layer
        # classifier.add(Dense(20, kernel_initializer='he_normal', activation = 'leaky_relu'))

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
                classifier.save('C:/Users/jhr96/Desktop/PYTHON/excel/HR')
            else:
                pass
        
        

        print('dense1 : %.1d / dense2 : %.1d Complete!!' %(dense1, dense2))

        #print(max(results.history['accuracy']))
        #print(max(results.history['val_accuracy']))
        #plt.plot(results.history['accuracy'])
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.show()  

        

       ####
        loss_and_metrics = classifier.evaluate(X_te, yyy_te, batch_size=1)
        #where = classifier.predict(X_te,batch_size=1)

        conclusion_2 = pd.DataFrame({'dense1': [dense1],
                                    'dense2': [dense2],
                                    'accuracy': [max(results.history['accuracy'])],
                                    'val_accuracy':[max(results.history['val_accuracy'])],
                                    'test_accuracy': [loss_and_metrics[1]]})

        conclusion = pd.concat([conclusion, conclusion_2], ignore_index=True)

        #print('')
        #print('loss_and_metrics : ' + str(loss_and_metrics))
        #print(str(where))
        #####

        #from keras.models import load_model
        #model = load_model("chan.h5")

        #model.evaluate(X_te,yyy_te, batch_size=1)
        #model.predict(X_te,batch_size=1)

conclusion.to_excel('conclusion.xlsx')


# conclusion 이용해서 제일 좋은 성능 나오는 dense1 dense2 를 빼오는 코딩 1줄
