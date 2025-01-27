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

from baseline_models import *

limit_gpus()
set_gpu(1)

dpi=100

dataset = "mnist"
ds_train, ds_test, ds_info = load_dataset(dataset, 128)

input_shape = ds_info.features['image'].shape
num_classes = ds_info.features['label'].num_classes

# x_train, y_train = get_raw_dat(ds_train)

xs, ys = next(iter(ds_train))

model = PhasorModel(input_shape, onehot_offset=0.0, onehot_phase=0.5, max_step=0.05, projection="NP")
model.compile(optimizer="rmsprop")

loss = model.train_with_sleep(ds_train, 2)