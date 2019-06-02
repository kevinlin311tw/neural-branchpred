import numpy as np
import os, sys
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Embedding
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model
import settings as s
from seq_predictors.predictor import Predictor
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from glob import glob
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
        # configure callbacks
        self.WEIGHTS_SAVE = './checkpoints/weights.{epoch:04d}.h5'
        self.TRAINING_LOG = './checkpoints/training_log.csv'
        checkpoint = ModelCheckpoint(self.WEIGHTS_SAVE, monitor='loss', verbose=0, save_best_only=False, mode='min', period=2)
        csv_logger = CSVLogger(self.TRAINING_LOG, append=True)
        callbacks_list = [checkpoint, csv_logger]

        self.model = Sequential()
        # max_features = 1000
        # self.model.add(Embedding(max_features, input_shape=(1,), output_dim=256))
        self.model.add(LSTM(128,input_shape=(max_seq_len,max_feat_len)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))


        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model.summary()
        self.model.fit(data, label, epochs=10, batch_size=10, callbacks=callbacks_list)    


    def get_last_epoch_and_weights_file(self):
        os.makedirs('checkpoints', exist_ok=True)
        # os.makedirs('checkpoint')
        files = [file for file in glob('./checkpoints/weights.*.h5')]
        files = [file.split('/')[-1] for file in files]
        epochs = [file.split('.')[1] for file in files if file]
        epochs = [int(epoch) for epoch in epochs if epoch.isdigit() ]
        if len(epochs) == 0:
            if 'weights.best.h5' in files:
                return -1, './checkpoints/weights.best.h5'
        else:
            ep = max([int(epoch) for epoch in epochs])
            return ep, self.WEIGHTS_SAVE.format(epoch=ep)
        return None, None

    def load_best_model(self):
        ep, modelfile = self.get_last_epoch_and_weights_file()
        self.model = load_model(modelfile)
         
    def predict(self, input_seq):
        # print("*** LSTM:", boxed_inst.shape)
        boxed_inst = np.expand_dims(input_seq, axis=0) 
        return self.model.predict(boxed_inst)
            
