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
from keras.layers import SimpleRNN, LSTM, Embedding
from keras.layers import Dense, Activation, Dropout

import settings as s
from predictors.predictor import Predictor

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
            int(d[s.PC], 16)
            #int(d[s.FALLTHROUGH], 16),
            #int(d[s.TARGET], 16)
            ]) for d in data])
        out = np.array([np.array([
            int(d[s.BRANCH] == 'T')
        ]) for d in data])
        self.model.fit(inp, out, epochs=100, batch_size=10)    
        
    def predict(self, inst):
        boxed_inst = np.array([
            int(inst[s.PC], 16)
            #int(inst[s.FALLTHROUGH], 16),
            #int(inst[s.TARGET], 16)
        ])
        if int(self.model.predict(boxed_inst)):
            return 'T'
        return 'N'
