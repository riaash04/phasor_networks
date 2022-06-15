import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as be

from utils import *
from data import *
from layers import *
from models import *

import os

#limit_gpus()
#set_gpu(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dataset = "mnist"

ds_train, ds_test, ds_info = load_dataset(dataset, 128)

input_shape = ds_info.features['image'].shape
num_classes = ds_info.features['label'].num_classes

# x_train, y_train = get_raw_dat(ds_train)

xs, ys = next(iter(ds_train))

sleep_acc = []
n_acc = []

xtest, ytest = ([], [])

#for i in [10, 50, 100, 200]:
#    model = PhasorModel(input_shape, onehot_offset=0.0, onehot_phase=0.5, max_step=0.05, projection="NP")
#    model.compile(optimizer="rmsprop")

#    model2 = PhasorModel(input_shape, onehot_offset=0.0, onehot_phase=0.5, max_step=0.05, projection="NP")
#    model2.compile(optimizer="rmsprop")

#    loss = model.train_with_sleep(ds_train, 2, i)
#    conf = model.accuracy(ds_test)
#    sleep_acc.append(confusion_to_accuracy(conf))
#    print(confusion_to_accuracy(conf))

#    loss2 = model2.train(ds_train, 2)
#    conf2 = model2.accuracy(ds_test)
#    n_acc.append(confusion_to_accuracy(conf2))
#    print(confusion_to_accuracy(conf2))

model3 = PhasorModel(input_shape, onehot_offset=0.0, onehot_phase=0.5, max_step=0.05, projection="NP")
model3.compile(optimizer="rmsprop")

loss3 = model3.train(ds_train, 2)
conf3 = model3.accuracy(ds_test)

print(confusion_to_accuracy(conf3))

#print("Sleep: ", sleep_acc)
#print("Normal: ", n_acc)

