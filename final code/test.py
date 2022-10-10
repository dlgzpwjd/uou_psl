import pandas as pd
import numpy as np
import os
import joblib
import keras
from keras.models import Sequential
from keras.models import load_model

keras.utils.set_random_seed(10)
model = Sequential()

dir = os.path.realpath(__file__)
path = os.path.abspath(os.path.join(dir, os.pardir))

data = pd.read_excel(path + "/dataset/Test DATA 1700.xlsx")
X = data.drop(['target','type','m'], axis = 1)
y = data.filter(['type'])
z = data.filter(['target'])
        
knn_test=joblib.load(path + '/model/knn.pkl')
knn_pred = knn_test.predict(X)
data['knn_pred'] = knn_pred

normal = data[data['knn_pred'] == 0]
ground = data[data['knn_pred'] == 1]
line = data[data['knn_pred'] == 2]
phase = data[data['knn_pred'] == 3]

normal_idx = []
gnd_idx = []
line_idx = []
ph_idx = []

for idx, row in normal.iterrows():
    normal_idx.append(idx)
for idx, row in ground.iterrows():
    gnd_idx.append(idx)
for idx, row in line.iterrows():
    line_idx.append(idx)
for idx, row in phase.iterrows():
    ph_idx.append(idx)

################################################ a 지락 test ################################################
X_g = ground.drop(['target', 'type', 'm', 'knn_pred'], axis = 1)

lda1 = joblib.load(path + '/model/lda1.pkl')
X_a = X_g.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am',
                  'IC_aph','VD_am','VD_aph','ID_am','ID_aph'])
X_a_lda = lda1.transform(X_a)
Xa_lda = pd.DataFrame(X_a_lda)

modela = load_model(path + '/model/ground_a.h5')
pred_a = modela.predict(Xa_lda, batch_size = 1)

################################################ b 지락 test ################################################
lda2 = joblib.load(path + '/model/lda2.pkl')
X_b = X_g.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm',
                  'IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
X_b_lda = lda2.transform(X_b)
Xb_lda = pd.DataFrame(X_b_lda)

modelb = load_model(path + '/model/ground_b.h5')
pred_b = modelb.predict(Xb_lda, batch_size = 1)

################################################ c 지락 test ################################################
lda3 = joblib.load(path + '/model/lda3.pkl')
X_c = X_g.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm',
                  'IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])
X_c_lda = lda3.transform(X_c)
Xc_lda = pd.DataFrame(X_c_lda)

modelc = load_model(path + '/model/ground_c.h5')
pred_c = modelc.predict(Xc_lda, batch_size = 1)

################################################ ab 단락 test ################################################
X_l = line.drop(['target', 'type', 'm', 'knn_pred'], axis = 1)

lda4 = joblib.load(path + '/model/lda4.pkl')
X_ab = X_l.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am',
                   'IC_aph','VD_am','VD_aph','ID_am','ID_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph',
                   'IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
X_ab_lda = lda4.transform(X_ab)
Xab_lda = pd.DataFrame(X_ab_lda)

modelab = load_model(path + '/model/line_ab.h5')
pred_ab = modelab.predict(Xab_lda, batch_size = 1)

################################################ bc 단락 test ################################################
lda5 = joblib.load(path + '/model/lda5.pkl')
X_bc = X_l.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VA_cm','VA_cph','IA_cm',
                   'IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph',
                   'ID_bm','ID_bph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])
X_bc_lda = lda5.transform(X_bc)
Xbc_lda = pd.DataFrame(X_bc_lda)

modelbc = load_model(path + '/model/line_bc.h5')
pred_bc = modelbc.predict(Xbc_lda, batch_size = 1)

################################################ ca 단락 test ################################################
lda6 = joblib.load(path + '/model/lda6.pkl')
X_ca = X_l.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm',
                   'IC_cph','VD_cm','VD_cph','ID_cm','ID_cph','VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph',
                   'IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])
X_ca_lda = lda6.transform(X_ca)
Xca_lda = pd.DataFrame(X_ca_lda)

modelca = load_model(path + '/model/line_ca.h5')
pred_ca = modelca.predict(Xca_lda, batch_size = 1)

################################################ 3상 단락 test ################################################
X_abc = phase.drop(['target', 'type', 'm', 'knn_pred'], axis = 1)

lda7 = joblib.load(path + '/model/lda7.pkl')
Xabc_lda = lda7.transform(X_abc)
Xabc = pd.DataFrame(Xabc_lda)

modelabc = load_model(path + '/model/3phase.h5')
pred_abc = modelabc.predict(Xabc, batch_size = 1)    

#################################################### Normal ####################################################
correct_n = 0

for i in range(0, len(normal)):
    if normal['type'].to_numpy()[i] == 0:
        correct_n = correct_n + 1

################################################ test 결과 확인 ################################################
pred_target = np.zeros((len(X), 1))
pred_type = np.zeros((len(X), 1))

pred_a = np.reshape(np.argmax(pred_a, axis=1), (-1, 1))
pred_b = np.reshape(np.argmax(pred_b, axis=1), (-1, 1))
pred_c = np.reshape(np.argmax(pred_c, axis=1), (-1, 1))

pred_g = np.concatenate((pred_a, pred_b, pred_c), axis=1)

i = 0
for idx in gnd_idx:
    pred_target[idx] = np.sum(pred_g[i, :])
    pred_type[idx] = np.argmax(pred_g[i, :]) + 1 
    i = i + 1

y_g = ground.filter(['target'])
z_g = ground.filter(['type'])

tar_g = np.sum(pred_g, axis = 1)
type_g = np.argmax(pred_g, axis = 1) + 1

correct_g = 0

for i in range(0, len(pred_g)):
    if tar_g[i] == y_g.to_numpy()[i] and type_g[i] == z_g.to_numpy()[i]:
        correct_g = correct_g + 1

accuracy_g = correct_g / 1000 * 100

print("지락 Accuracy : %.4f %%" %(accuracy_g))
###############################################################################
pred_ab = np.reshape(np.argmax(pred_ab, axis=1), (-1, 1))
pred_bc = np.reshape(np.argmax(pred_bc, axis=1), (-1, 1))
pred_ca = np.reshape(np.argmax(pred_ca, axis=1), (-1, 1))

pred_l = np.concatenate((pred_ab, pred_bc, pred_ca), axis=1)

i = 0
for idx in line_idx:
    pred_target[idx] = np.sum(pred_l[i, :])
    pred_type[idx] = np.argmax(pred_l[i, :]) + 4 
    i = i+1

y_l = line.filter(['target'])
z_l = line.filter(['type'])

tar_l = np.sum(pred_l, axis = 1)
type_l = np.argmax(pred_l, axis = 1) + 4

correct_l = 0

for i in range(0, len(pred_l)):
    if tar_l[i] == y_l.to_numpy()[i] and type_l[i] == z_l.to_numpy()[i]:
        correct_l = correct_l + 1
        

accuracy_l = correct_l / 300 * 100

print("선간 Accuracy : %.4f %%" %(accuracy_l))
###############################################################################
pred_p = np.reshape(np.argmax(pred_abc, axis=1), (-1, 1))

i = 0
for idx in ph_idx:
    pred_target[idx] = pred_p[i]
    pred_type[idx] = 7
    i = i+1

data['pred_target'] = pred_target + 1
data['pred_type'] = pred_type

y_p = phase.filter(['target']) - 1
z_p = phase.filter(['type'])

correct_p = 0

for i in range(0, len(pred_p)):
    if z_p.to_numpy()[i] == 7 and pred_p[i] == y_p.to_numpy()[i]:
        correct_p = correct_p + 1

accuracy_p = correct_p / 100 * 100

print("3상 Accuracy : %.4f %%" %(accuracy_p))

data.to_excel(path + "/result.xlsx")

accuracy_total = (correct_p + correct_l + correct_n + correct_g) / 1700 *100
print("total Accuracy : %.4f %%" %(accuracy_total))