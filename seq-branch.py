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
import random
def address_to_nhot(input):
    data_int = int(input,16)+1
    # return [int(i) for i in list('{0:0b}'.format(data_int))] 
    output = [0]*23
    count = 0
    for i in bin(data_int)[2:]:
        output[count] = int(i)
        count+=1
    return output

def feat_extract(inst_seq):
    feat_seq = []
    for inst in inst_seq:
        inst = inst.split()
        pc_vec = address_to_nhot(inst[s.PC])
        f_vec = address_to_nhot(inst[s.FALLTHROUGH])
        t_vec = address_to_nhot(inst[s.TARGET])
        src1_vec = address_to_nhot(inst[s.SRC1])
        src2_vec = address_to_nhot(inst[s.SRC2])
        dst_vec = address_to_nhot(inst[s.DEST])
        # concatenate all lists
        concat_vec = pc_vec + f_vec + t_vec + src1_vec + src2_vec + dst_vec
        # concat_vec = list(np.concatenate([pc_vec,f_vec,t_vec]))
        # code.interact(local=locals())
        
        feat_seq.append(concat_vec)
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
    # load checkpoints
    predictor.load_best_model()
    for i in range(len(data)):
        target = label[i]
        pred = predictor.predict(data[i])
        print('pred prob = %f, pred = %d, gt = %d'%(pred,(pred>0.5),target))
        correct += int(target==(pred>0.5))
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
    # code.interact(local=locals()) 
    # data2 = random.Random(123).shuffle(data)
    # label2 = random.Random(123).shuffle(label)
    # code.interact(local=locals())
    data_split = np.array_split(data, 5)
    label_split = np.array_split(label, 5)
    test_data  = np.concatenate([data_split[1],data_split[3]])
    train_data  = np.concatenate([data_split[0],data_split[2],data_split[4]])

    test_label  = np.concatenate([label_split[1],label_split[3]])
    train_label  = np.concatenate([label_split[0],label_split[2],label_split[4]])

    # code.interact(local=locals())   
    tests = {
	"neural LSTM (ours)"  : NeuralLSTMPredictor(train_data, train_label),
    }
    
    for predictor in tests:
        print("{} predictor had {} accuracy".format(
            predictor, evaluate(tests[predictor], test_data, test_label)))
   
     
if __name__ == "__main__":
    main(filename="data/gcc-10M_seq.trace")
