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
from keras.layers import Dense, Activation, Dropout

import settings as s
from predictors.predictor import Predictor
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
import code
from textwrap import wrap

class NeuralPredictor(Predictor):
    def __init__(self, data):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu'))
        '''
        self.model.add(Dropout(0.5)) 
        self.model.add(Dense(32, activation='relu', input_dim=1))
        self.model.add(Dropout(0.5)) 
        self.model.add(Dense(32, activation='relu', input_dim=1))
        self.model.add(Dropout(0.5)) 
        self.model.add(Dense(32, activation='relu', input_dim=1))
        self.model.add(Dropout(0.5)) 
        '''
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
         
        traindata = []
        for d in data:
            pc_data = wrap(bin(int(d[s.PC],16))[2:],1)
            binary_pc = [float(i) for i in pc_data]
            f_data = wrap(bin(int(d[s.FALLTHROUGH],16))[2:],1)
            binary_f = [float(i) for i in f_data]
            t_data = wrap(bin(int(d[s.TARGET],16))[2:],1)
            binary_t = [float(i) for i in t_data]
            cat_feat = np.concatenate([binary_pc, binary_f, binary_t])
            traindata.append(np.array(cat_feat))
        inp = np.array(traindata)
        '''
        inp = np.array([np.array([
            int(d[s.PC],16),
            # int(d[s.FALLTHROUGH], 16),
            # int(d[s.TARGET], 16)%4730000,
            ]) for d in data])
        ''' 
        out = np.array([np.array([
            int(d[s.BRANCH] == 'T')
        ]) for d in data])
        # code.interact(local=locals())
        print("** NN: inp dim:{}".format(inp.shape))
        print("** NN: out dim:{}".format(out.shape))   
        
        self.model.fit(inp, out, epochs=100, batch_size=10) 
        print("** NN: inp dim:{}".format(inp.shape))
        print("** NN: out dim:{}".format(out.shape))   
        # print('finish training')        
    def predict(self, inst):
        
        pc_data = wrap(bin(int(inst[s.PC],16))[2:],1)
        binary_pc = [float(i) for i in pc_data]
        f_data = wrap(bin(int(inst[s.FALLTHROUGH],16))[2:],1)
        binary_f = [float(i) for i in f_data]
        t_data = wrap(bin(int(inst[s.TARGET],16))[2:],1)
        binary_t = [float(i) for i in t_data]
        cat_feat = np.concatenate([binary_pc, binary_f, binary_t])
        boxed_inst = np.array(cat_feat)
        
        # pc_data = wrap(bin(int(inst[s.PC],16))[2:],1)
        # binary_pc = [float(i) for i in pc_data]
        # boxed_inst = np.array(binary_pc)
        
        '''
        boxed_inst = np.array([
            int(inst[s.PC], 16),
            # int(inst[s.FALLTHROUGH], 16),
            # int(inst[s.TARGET], 16)
        ])
        ''' 
        # print("**** NN:",boxed_inst.shape)
        boxed_inst = np.expand_dims(boxed_inst, axis=0)
        if int(self.model.predict(boxed_inst)):
            return 'T'
        return 'N'
