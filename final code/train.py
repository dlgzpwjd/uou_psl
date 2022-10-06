import os

from traincode.knn_train import * 
from traincode.line_fault_train import *
from traincode.ground_fault_train import *
from traincode.th_phase_fault_train import *

epochs = 200

dir = os.path.realpath(__file__)
dataset_path = os.path.abspath(os.path.join(dir, os.pardir))

train_knn(dataset_path)
train_gnd(dataset_path, epochs)
train_line(dataset_path, epochs)
train_3ph(dataset_path, epochs)