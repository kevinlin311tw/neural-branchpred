import timeit
from timeit import default_timer as timer

SETUP_CODE = '''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from seq_branch import preprocess
from seq_predictors.neuralLSTM  import NeuralLSTMPredictor
from keras.preprocessing import sequence
    
filename = "data/gcc-10M_seq.trace"
data, label = preprocess(filename)
data = sequence.pad_sequences(data, maxlen=40)
i = int(data.shape[0]/1000)
test_data   = data[:i]
test_label  = label[:i]
train_data  = data[i:]
train_label = label[i:]
print("TEST DATA shape: {}".format(test_data.shape))

predictor = NeuralLSTMPredictor(train_data, train_label)
predictor.load_best_model()
'''

TEST_CODE = '''
for i in range(len(test_data)):
    pred = predictor.predict(test_data[i])
'''
start = timer()
times = timeit.timeit(setup = SETUP_CODE, stmt = TEST_CODE, number=1)
end = timer()
print("time elapsed: {}".format(end - start))
with open("LSTM_latency.log", mode="w") as f:
   f.write("LSTM took time {} to predict".format(times))
