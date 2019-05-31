"""
__name__ = neural.py
__author__ = Yash Patel
__description__ = Neural branch predictor, which was of original
interest and to be compared to these other class benchmarks. Here
it is only presented as a simply preceptron unit, though we expand
into networks and reinforcement learning as well
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

import settings as s
from predictors.predictor import Predictor
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



class NeuralPredictor(Predictor):
    def __init__(self, data):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_dim=3))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        inp = np.array([np.array([
            int(d[s.PC], 16),
            int(d[s.FALLTHROUGH], 16),
            int(d[s.TARGET], 16),
            ]) for d in data])
        out = np.array([np.array([
            int(d[s.BRANCH] == 'T')
        ]) for d in data])
        self.model.fit(inp, out, epochs=10, batch_size=10) 
        print("** NN: inp dim:{}".format(inp.shape))
        print("** NN: out dim:{}".format(out.shape))   
        print('finish training')        
    def predict(self, inst):
        boxed_inst = np.array([
            int(inst[s.PC], 16),
            int(inst[s.FALLTHROUGH], 16),
            int(inst[s.TARGET], 16)
        ])
        print("**** NN:",boxed_inst.shape)
        boxed_inst = np.expand_dims(boxed_inst, axis=0)
        if int(self.model.predict(boxed_inst)):
            return 'T'
        return 'N'
