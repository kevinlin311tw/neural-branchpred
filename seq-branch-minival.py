"""
__description__ = Branch prediction algorithm (simple non-neural
implementation) used for benchmarking. Visualization is handled
separately from the processing here
"""

import numpy as np
from keras.preprocessing import sequence
from seq_predictors.neuralLSTM  import NeuralLSTMPredictor
from visualization.dynamic import visualize_test
import settings as s
import json
import code
from textwrap import wrap
from tqdm import tqdm

def feat_extract(inst_seq):
    feat_seq = []
    for inst in inst_seq:
        inst = inst.split()
        pc_int = int(inst[s.PC],16)
        # pc_bin = bin(pc_int)[2:]
        # pc_bin = wrap(pc_bin,1)
        # pc_vector = [float(i) for i in pc_bin]
        pc_vector = [int(i) for i in list('{0:0b}'.format(pc_int))] 
        feat_seq.append(pc_vector)
    return np.array(feat_seq)

def preprocess(filename):
    data = []
    label = []
    f = open(filename, "r")
    jsondata = json.load(f) 
    print('start loading data + feature extraction')
    for item in tqdm(jsondata):
        seq_feat = feat_extract(item['data'])
        seq_label = item['label']
        data.append(seq_feat) 
        label.append(seq_label)
    return np.array(data), np.array(label)

def get_max_len(data):
    inst_len = []
    for item in data:
        inst_len.append(item.shape[0])
    return np.max(inst_len)


def evaluate(predictor, data, label):
    """
    Given a predictor, as defined in the predictor directory (either the
    static predictor, dynamic, or neural) calculates the accuracy through
    the dump provided and outputs accuracy (as percent)
    """
    correct = 0
    for i in range(len(data)):
        target = label[i]
        pred = predictor.predict(data[i])
        correct += int(target==pred)
    return correct/len(data)

def main(filename):
    data, label = preprocess(filename)
    print('loaded data shape:',data.shape)
    print('loaded label shape:',label.shape)
    print('max inst len: %d'%(get_max_len(data)))
    max_length = 40  # 40 for 10M dataset, 21 for 1K dataset
    data = sequence.pad_sequences(data, maxlen=max_length)
    # code.interact(local=locals())
    # part of the dump corresponding to static training "history"
    # data that is not seen live by user
    data_split = np.array_split(data, 5)
    test_data = data_split[0]
    train_data  = np.concatenate(data_split[1:])
    label_split = np.array_split(label, 5)
    test_label = label_split[0]
    train_label  = np.concatenate(label_split[1:])
    # code.interact(local=locals())   
    tests = {
	"neural LSTM (ours)"  : NeuralLSTMPredictor(train_data, train_label),
    }

    for predictor in tests:
        print("{} predictor had {} accuracy".format(
            predictor, evaluate(tests[predictor], test_data, test_label)))
   
     
if __name__ == "__main__":
    main(filename="data/gcc-1K_seq.trace")
