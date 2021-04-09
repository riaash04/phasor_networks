import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from utils import *
from scipy.integrate import solve_ivp
from tqdm import tqdm

class StaticLinear(keras.layers.Layer):
    def __init__(self, n_in, n_out, overscan=1.0):
        super(StaticLinear, self).__init__()
        
        self.w = tf.Variable(
            initial_value = construct_sparse(n_in, n_out, overscan),
            trainable = False)

        self.rev_w = tf.linalg.pinv(self.w)

    def reverse(self, inputs):
        return tf.matmul(inputs, self.rev_w)
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w)

class CmpxLinear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CmpxLinear, self).__init__()
        self.units = units
        #dynamic execution constants
        self.leakage = kwargs.get("leakage", -0.2)
        self.period = kwargs.get("period", 1.0)
        #set the eigenfrequency to 1/T
        self.ang_freq = 2 * np.pi / self.period
        self.window = kwargs.get("window", 0.05)
        self.spk_mode = kwargs.get("spk_mode", "gradient")
        self.threshold = kwargs.get("threshold", 0.03)
        self.exec_time = kwargs.get("exec_time", 10.0)
        self.max_step = kwargs.get("max_step", 0.005)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="w"
        )
        self.n_in = self.w.shape[0]

        #add bias?
        # self.b = self.add_weight(
        #     shape=self.units,
        #     initializer="zeros",
        #     trainable=True
        # )

    def call(self, inputs, mode="static"):
        if mode=="dynamic":
            output = self.call_dynamic(inputs)
        else:
            output = self.call_static(inputs)

        return output

    def current(self, t, spikes):
        spikes_i, spikes_t = spikes
        window = self.window
        neurons = self.n_in
        currents = np.zeros((neurons), dtype="float")

        box_start = t - window
        box_end = t + window
        active = (spikes_t > box_start) * (spikes_t < box_end)
        active_i = np.unique(spikes_i[active])
        for i in active_i:
            currents[i] += 1.0

        currents = tf.constant(currents, dtype="float")
        currents = tf.reshape(currents, (1,-1))
        return currents

    def dz(self, t, z, current):
        k = tf.complex(self.leakage, self.ang_freq)
        
        #scale currents by synaptic weight
        currents = tf.matmul(current(t), self.w)
        currents = tf.complex(currents, tf.zeros_like(currents))
        
        dz = k * z + currents
        return dz.numpy()

    #currently inference only
    def call_dynamic(self, inputs):
        solutions = []
        outputs = []
        n_batches = len(inputs)

        for i in tqdm(range(n_batches)):
            input_i = inputs[i][0]
            input_t = inputs[i][1]
            z0 = np.zeros((self.units), "complex")

            i_fn = lambda t: self.current(t, (input_i, input_t))
            dz_fn = lambda t,z: self.dz(t, z, i_fn)
            sol = solve_ivp(dz_fn, (0.0, self.exec_time), z0, max_step=self.max_step)
            solutions.append(sol)

            if self.spk_mode == "gradient":
                spk = findspks(sol, threshold=self.threshold, period=self.period)
            elif self.spk_mode == "cyclemax":
                spk = findspks_max(sol, threshold=self.threshold, period=self.period)
            else:
                print("WARNING: Spike mode not recognized, defaulting to gradient")
                spk = findspks(sol, threshold=self.threshold, period=self.period)
            
            spk_inds, spk_tms = np.nonzero(spk)
            spk_tms = sol.t[spk_tms]

            outputs.append( (spk_inds, spk_tms) )

        self.solutions = solutions
        self.spike_trains = outputs
        return outputs


    def call_static(self, inputs):
        pi = tf.constant(np.pi)
        #clip inputs to -1, 1 domain (pi-normalized phasor)
        inputs = tf.clip_by_value(inputs, -1, 1)
        #convert the phase angles into complex vectors
        inputs = phase_to_complex(inputs)
        #scale the complex vectors by weight and sum
        inputs = tf.matmul(inputs, tf.complex(self.w, tf.zeros_like(self.w)))
        #convert them back to phase angles
        output = tf.math.angle(inputs) / pi

        return output


    def get_config(self):
        config = super(CmpxLinear, self).get_config()
        config.update({"units": self.units})
        config.update({"w": self.w.numpy()})
        return config

    def get_weights(self):
        return [self.w.numpy()]

    def set_weights(self, weights):
        self.w.value = weights[0]
        

class Normalize(keras.layers.Layer):
    def __init__(self, sigma, **kwargs):
        super(Normalize, self).__init__()

        self.sigma = tf.constant(sigma)

        momentum = kwargs.get("momentum", 0.99)
        self.momentum = tf.constant(momentum)
        
        epsilon = kwargs.get("epsilon", 0.001)
        self.epsilon = tf.constant(
            epsilon * tf.ones((1,), dtype="float32")
        )

        self.moving_mean = tf.Variable(
            initial_value = tf.zeros((1,), dtype="float32"),
            trainable = False
        )

        self.moving_std = tf.Variable(
            initial_value = tf.ones((1,), dtype="float32"),
            trainable = False
        )

    def call(self, data, **kwargs):
        training = kwargs.get("training", True)

        if training:
            #calculate batch moments
            mean, var = tf.stop_gradient(tf.nn.moments(tf.reshape(data, -1),axes=0))
            std = tf.math.sqrt(var)

            #updating the moving moments
            self.moving_mean = self.moving_mean * self.momentum + mean * (1-self.momentum)
            self.moving_std = self.moving_std * self.momentum + std * (1-self.momentum)

            #batchnorm scaling
            output = (data - mean) / (std + self.epsilon) 
            return output / self.sigma

        else:
            #scale with calculated moments
            mean = self.moving_mean
            std = self.moving_std

            #batchnorm scaling
            output = (data - mean) / (std + self.epsilon)
            return output / self.sigma

    def reverse(self, data):
        #undo scaling with calculated moments
        mean = self.moving_mean
        std = self.moving_std

        x = self.sigma * data
        output = (x)*(std + self.epsilon) + mean
        
        return output