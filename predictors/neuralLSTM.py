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


class NeuralLSTMPredictor(Predictor):
    def __init__(self, data):
        self.model = Sequential()
        max_features = 32
        self.model.add(Embedding(max_features, output_dim=256))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))


        self.model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        inp = np.array([np.array([
            int(d[s.PC], 16),
            int(d[s.FALLTHROUGH], 16),
            int(d[s.TARGET], 16)
            ]) for d in data])
        out = np.array([np.array([
            int(d[s.BRANCH] == 'T')
        ]) for d in data])
        self.model.fit(inp, out, epochs=10, batch_size=10)    
        
    def predict(self, inst):
        boxed_inst = np.array([
            int(inst[s.PC], 16),
            int(inst[s.FALLTHROUGH], 16),
            int(inst[s.TARGET], 16)
        ])
        print("*** LSTM:", boxed_inst.shape)
        if int(self.model.predict(boxed_inst)):
            return 'T'
        return 'N'
