import pandas as pd
import numpy as np
import joblib
import keras
from keras.models import Sequential
from keras.models import load_model

keras.utils.set_random_seed(1)
model = Sequential()

data = pd.read_excel("C:/Users/user/Desktop/psl/Test DATA 1700.xlsx")
X = data.drop(['target','type','m'], axis = 1)
y = data.filter(['type'])
z = data.filter(['target'])
# target = []

# for idx, row in data.iterrows():
#     if row['type'] == 0:
#         target.append(0)
#     elif row['type'] == 1 or row['type'] == 2 or row['type'] == 3:
#         target.append(1)
#     elif row['type'] == 4 or row['type'] == 5 or row['type'] == 6:
#         target.append(2)
#     else:
#         target.append(3)
        
knn_test=joblib.load('./knn_model.pkl')
knn_pred = knn_test.predict(X)
data['knn_pred'] = knn_pred

# accuracy = accuracy_score(target, knn_pred)
# print(accuracy)

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

lda1 = joblib.load('./lda1.pkl')
X_a = X_g.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am',
                  'IC_aph','VD_am','VD_aph','ID_am','ID_aph'])
X_a_lda = lda1.transform(X_a)
Xa_lda = pd.DataFrame(X_a_lda)

modela = load_model("./ground_a.h5")
pred_a = modela.predict(Xa_lda, batch_size = 1)

################################################ b 지락 test ################################################
lda2 = joblib.load('./lda2.pkl')
X_b = X_g.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm',
                  'IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
X_b_lda = lda2.transform(X_b)
Xb_lda = pd.DataFrame(X_b_lda)

modelb = load_model("./ground_b.h5")
pred_b = modelb.predict(Xb_lda, batch_size = 1)

################################################ c 지락 test ################################################
lda3 = joblib.load('./lda3.pkl')
X_c = X_g.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm',
                  'IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])
X_c_lda = lda3.transform(X_c)
Xc_lda = pd.DataFrame(X_c_lda)

modelc = load_model("./ground_c.h5")
pred_c = modelc.predict(Xc_lda, batch_size = 1)

################################################ ab 단락 test ################################################
X_l = line.drop(['target', 'type', 'm', 'knn_pred'], axis = 1)

lda4 = joblib.load('./lda4.pkl')
X_ab = X_l.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am',
                   'IC_aph','VD_am','VD_aph','ID_am','ID_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph',
                   'IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
X_ab_lda = lda4.transform(X_ab)
Xab_lda = pd.DataFrame(X_ab_lda)

modelab = load_model("./line_ab.h5")
pred_ab = modelab.predict(Xab_lda, batch_size = 1)

################################################ bc 단락 test ################################################
lda5 = joblib.load('./lda5.pkl')
X_bc = X_l.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VA_cm','VA_cph','IA_cm',
                   'IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph',
                   'ID_bm','ID_bph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])
X_bc_lda = lda5.transform(X_bc)
Xbc_lda = pd.DataFrame(X_bc_lda)

modelbc = load_model("./line_bc.h5")
pred_bc = modelbc.predict(Xbc_lda, batch_size = 1)

################################################ ca 단락 test ################################################
lda6 = joblib.load('./lda6.pkl')
X_ca = X_l.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm',
                   'IC_cph','VD_cm','VD_cph','ID_cm','ID_cph','VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph',
                   'IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])
X_ca_lda = lda6.transform(X_ca)
Xca_lda = pd.DataFrame(X_ca_lda)

modelca = load_model("./line_ca.h5")
pred_ca = modelca.predict(Xca_lda, batch_size = 1)

################################################ 3상 단락 test ################################################
X_abc = phase.drop(['target', 'type', 'm', 'knn_pred'], axis = 1)

lda7 = joblib.load('./lda7.pkl')
Xabc_lda = lda7.transform(X_abc)
Xabc = pd.DataFrame(Xabc_lda)

modelabc = load_model("./3phase.h5")
pred_abc = modelabc.predict(Xabc, batch_size = 1)    

##############################################################################################################

pred_target = np.zeros((len(X), 1))
pred_type = np.zeros((len(X), 1))

pred_a = np.reshape(np.argmax(pred_a, axis=1), (-1, 1))
pred_b = np.reshape(np.argmax(pred_b, axis=1), (-1, 1))
pred_c = np.reshape(np.argmax(pred_c, axis=1), (-1, 1))

pred_g = np.concatenate((pred_a, pred_b, pred_c), axis=1)

i = 0
for idx in gnd_idx:
    pred_target[idx] = np.sum(pred_g[i, :])
    pred_type[idx] = np.argmax(pred_g[i, :]) + 1 # 0 1 2 -> +1 을 해주면 1 2 3
    i = i+1

# pred_g = pd.DataFrame(pred_g)
# print(pred_g)
# pred_g.to_excel('result_ground.xlsx')

y_g = ground.filter(['target'])
z_g = ground.filter(['type'])
row_g = y_g.shape[0]
yy_g = y_g.to_numpy()
z_g = z_g.to_numpy()

con_g = np.zeros((row_g, 3))
for i in range(0, row_g-1):
    if z_g[i] == 1:
        con_g[i, 0] = yy_g[i] -1
    elif z_g[i] == 2:
        con_g[i, 1] = yy_g[i] -1
    elif z_g[i] == 3:
        con_g[i, 2] = yy_g[i] -1
 
# print(con_g)
last_g = (con_g == pred_g)
# print(last_g)

accuracy_g = np.sum(last_g) / (last_g.shape[0] * last_g.shape[1]) * 100

print("Accuracy : %.4f %%" %(accuracy_g))
###############################################################################
pred_ab = np.reshape(np.argmax(pred_ab, axis=1), (-1, 1))
pred_bc = np.reshape(np.argmax(pred_bc, axis=1), (-1, 1))
pred_ca = np.reshape(np.argmax(pred_ca, axis=1), (-1, 1))

pred_l = np.concatenate((pred_ab, pred_bc, pred_ca), axis=1)

i = 0
for idx in line_idx:
    pred_target[idx] = np.sum(pred_l[i, :])
    pred_type[idx] = np.argmax(pred_l[i, :]) + 4 # 0 1 2 -> +1 을 해주면 1 2 3
    i = i+1
# pred_l = pd.DataFrame(pred_l)
# print(pred_l.shape)
# pred_l.to_excel('result_line.xlsx')
y_l = line.filter(['target'])
z_l = line.filter(['type'])
row_l = y_l.shape[0]
yy_l = y_l.to_numpy()
z_l = z_l.to_numpy()

con_l = np.zeros((row_l, 3))
for i in range(0, row_l-1):
    if z_l[i] == 4:
        con_l[i, 0] = yy_l[i] -1
    elif z_l[i] == 5:
        con_l[i, 1] = yy_l[i] -1
    elif z_l[i] == 6:
        con_l[i, 2] = yy_l[i] -1
 
# print(con_l)
last_l = (con_l == pred_l)
# print(last_l)

accuracy_l = np.sum(last_l) / (last_l.shape[0] * last_l.shape[1]) * 100

print("Accuracy : %.4f %%" %(accuracy_l))
###############################################################################
pred_p = np.reshape(np.argmax(pred_abc, axis=1), (-1, 1))

i = 0
for idx in ph_idx:
    pred_target[idx] = np.sum(pred_p[i, :])
    pred_type[idx] = 7 # 0 1 2 -> +1 을 해주면 1 2 3
    i = i+1

data['pred_target'] = pred_target
data['pred_type'] = pred_type

y_p = phase.filter(['target']) -1
z_p = phase.filter(['type'])
con_p = y_p.to_numpy()
# print(con_p)
last_p = (con_p == pred_p)
# print(last_p)

accuracy_p = np.sum(last_p) / (last_p.shape[0] * last_p.shape[1]) * 100

print("Accuracy : %.4f %%" %(accuracy_p))
# pred_p = pd.DataFrame(pred_p)
# print(pred_p.shape)
data.to_excel('result.xlsx')

last_type = (y == pred_type)
accuracy_type = np.sum(last_type) / (last_type.shape[0] * last_type.shape[1]) * 100

print("type Accuracy : %.4f %%" %(accuracy_type))

last_target = (z == (pred_target+1))
accuracy_target = np.sum(last_target) / (last_target.shape[0] * last_target.shape[1]) * 100

print("target Accuracy : %.4f %%" %(accuracy_target))
