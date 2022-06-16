import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as be
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds

from data import *
from layers import *
from models import *
from utils import *

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

labels = [0, 1, 2, 3]
x_train_X = []
y_train_X = []

for _, data in enumerate(ds_train):
    x, y = data

    # print(tf.shape(x))
    # print(tf.shape(y))
    
    inds = tf.where([k in labels for k in y])[:, 0]
    x_train = tf.gather(x, inds, axis=0)
    y_train = tf.gather(y, inds, axis=0)

    if(np.shape(x_train_X)[0] == 0):
        x_train_X = x_train
        y_train_X = y_train
    else:
        x_train_X = tf.concat((x_train_X, x_train), axis=0)
        y_train_X = tf.concat((y_train_X, y_train), axis=0)

print(np.shape(x_train_X))
print(np.shape(y_train_X))
# ind1 = tf.where([k == 0 for k in y_train_X])[:, 0]
# x_train_F = tf.gather(x_train_X, ind1, axis=0)
# y_train_F = tf.gather(y_train_X, ind1, axis=0)

# for i in range(1, 4):
#     print(i)
#     ind1 = tf.where([k == i for k in y_train_X])[:, 0]
#     x_train_F = tf.concat((x_train_F, tf.gather(x_train_X, ind1, axis=0)), axis=0)
#     y_train_F = tf.concat((y_train_F, tf.gather(y_train_X, ind1, axis=0)), axis=0)

# print("F:", np.shape(x_train_F))
# print(np.shape(y_train_F))
filtered_ds_train_x = tf.data.Dataset.from_tensor_slices((x_train_X, y_train_X)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

labels = [0, 1, 2, 3]
x_test_X = []
y_test_X = []

for _, data in enumerate(ds_test):
    x, y = data

    # print(tf.shape(x))
    # print(tf.shape(y))
    
    inds = tf.where([k in labels for k in y])[:, 0]
    x_test = tf.gather(x, inds, axis=0)
    y_test = tf.gather(y, inds, axis=0)

    if(np.shape(x_test_X)[0] == 0):
        x_test_X = x_test
        y_test_X = y_test
    else:
        x_test_X = tf.concat((x_test_X, x_test), axis=0)
        y_test_X = tf.concat((y_test_X, y_test), axis=0)

print(np.shape(x_test_X))
print(np.shape(y_test_X))
filtered_ds_test_x = tf.data.Dataset.from_tensor_slices((x_test_X, y_test_X)).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

labels_to_train = [[0,1], [2,3]]
labels_to_sleep = [[0,1], [0,1,2,3]]
for i in [10, 50, 100, 500, 1000]:
    model = PhasorModel(input_shape, onehot_offset=0.0, onehot_phase=0.5, max_step=0.05, projection="NP")
    model.compile(optimizer="rmsprop")

    model2 = PhasorModel(input_shape, onehot_offset=0.0, onehot_phase=0.5, max_step=0.05, projection="NP")
    model2.compile(optimizer="rmsprop")
    for j in range(2):
        loss = model.train_with_sleep(filtered_ds_train_x, 1, labels_to_train[j], labels_to_sleep[j], i)
        conf = model.accuracy(filtered_ds_test_x)
        sleep_acc.append(confusion_to_accuracy(conf))
        print(confusion_to_accuracy(conf))

        loss2 = model2.train(filtered_ds_train_x, 1, labels_to_train[j])
        conf2 = model2.accuracy(filtered_ds_test_x)
        n_acc.append(confusion_to_accuracy(conf2))
        print(confusion_to_accuracy(conf2))

# ds_train_x = ds_train.enumerate()  # Create index,value pairs in the dataset.
# ds_test_x = ds_test.enumerate()

# # Create filter function:
# def filter_fn(idx, data):
#     x, y = data
#     return y in [0, 1, 2 , 3]

# # The above is not going to work in graph mode
# # We are wrapping it with py_function to execute it eagerly
# def py_function_filter(idx, data):
#     return tf.py_function(filter_fn, (idx, data), tf.bool)

# # Filter the dataset as usual:
# filtered_ds_train_x = ds_train_x.filter(py_function_filter)
# filtered_ds_test_x = ds_test_x.filter(py_function_filter)

# model3 = PhasorModel(input_shape, onehot_offset=0.0, onehot_phase=0.5, max_step=0.05, projection="NP")
# model3.compile(optimizer="rmsprop")

# print(ds_train)
# print(filtered_ds_train_x)

# loss3 = model3.train(filtered_ds_train_x, 2)
# conf3 = model3.accuracy(filtered_ds_test_x)

# print(np.shape(conf3))
# print(confusion_to_accuracy(conf3))

print("Sleep: ", sleep_acc)
print("Normal: ", n_acc)

