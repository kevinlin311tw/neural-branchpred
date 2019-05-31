import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Embedding
from keras.layers import Dense, Activation, Dropout

import settings as s
from predictors.predictor import Predictor
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


class NeuralStackedLSTMPredictor(Predictor):
    def __init__(self, data):
        self.model = Sequential()
        max_features = 32
        data_dim = 16
        timesteps = 8
        self.model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
        self.model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
        self.model.add(LSTM(32))  # return a single vector of dimension 32
        self.model.add(Dense(1, activation='softmax'))

        self.model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        inp = np.array([np.array([
            int(d[s.PC], 16)
            #int(d[s.FALLTHROUGH], 16),
            #int(d[s.TARGET], 16)
            ]) for d in data])
        out = np.array([np.array([
            int(d[s.BRANCH] == 'T')
        ]) for d in data])
        self.model.fit(inp, out, epochs=10, batch_size=10)    
        
    def predict(self, inst):
        boxed_inst = np.array([
            int(inst[s.PC], 16)
            #int(inst[s.FALLTHROUGH], 16),
            #int(inst[s.TARGET], 16)
        ])
        if int(self.model.predict(boxed_inst)):
            return 'T'
        return 'N'
