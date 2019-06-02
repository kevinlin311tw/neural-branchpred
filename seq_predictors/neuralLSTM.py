import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Embedding
from keras.layers import Dense, Activation, Dropout

import settings as s
from seq_predictors.predictor import Predictor
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
import code

class NeuralLSTMPredictor(Predictor):
    def __init__(self, data, label):
        batch_size = data.shape[0]
        max_seq_len = data.shape[1]
        max_feat_len = data.shape[2]

        self.model = Sequential()
        # max_features = 1000
        # self.model.add(Embedding(max_features, input_shape=(1,), output_dim=256))
        self.model.add(LSTM(128,input_shape=(max_seq_len,max_feat_len)))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))


        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model.summary()
        self.model.fit(data, label, epochs=20, batch_size=10)    
        
    def predict(self, input_seq):
       
        # print("*** LSTM:", boxed_inst.shape)
        boxed_inst = np.expand_dims(input_seq, axis=0) 
        return self.model.predict(boxed_inst)
            
