from traincode.knn_train import * 
from traincode.line_fault_train import *
from traincode.ground_fault_train import *
from traincode.th_phase_fault_train import *

epochs = 200

train_knn()
train_gnd(epochs)
train_line(epochs)
train_3ph(epochs)