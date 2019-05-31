"""
__name__ = branch.py
__author__ = Yash Patel
__description__ = Branch prediction algorithm (simple non-neural
implementation) used for benchmarking. Visualization is handled
separately from the processing here
"""

import numpy as np

from predictors.static  import StaticPredictor
from predictors.bimodal import BimodalPredictor
from predictors.gshare  import GSharePredictor
from predictors.neural  import NeuralPredictor
from predictors.neuralLSTM  import NeuralLSTMPredictor

from visualization.dynamic import visualize_test
import settings as s

def preprocess(filename):
    """
    Preprocesses the file at the specified name and returns only
    the branching (conditional) lines in the dump, defined to have
    (in the format specified for the offline dump versions):
    1) Reads the flags register (that is, conditionRegister == 'R'), and
    2) Is either taken or not taken (that is, TNnotBranch != '-').
    """
    cleaned = []
    with open(filename, "r") as f:
        for line in f:
            instruction_dump = line.split()
            if instruction_dump[s.FLAGS] == 'R' and \
               instruction_dump[s.BRANCH] != '-':
               cleaned.append(instruction_dump)
    return np.array(cleaned)

def evaluate(predictor, data):
    """
    Given a predictor, as defined in the predictor directory (either the
    static predictor, dynamic, or neural) calculates the accuracy through
    the dump provided and outputs accuracy (as percent)
    """
    correct = 0
    for inst in data:
        correct += int(inst[s.BRANCH] == predictor.predict(inst))
    return correct/len(data)

def main(filename):
    memdump = preprocess(filename)
    # part of the dump corresponding to static training "history"
    # data that is not seen live by user
    split = np.array_split(memdump, 5)
    traindump = split[0]
    testdump  = np.concatenate(split[1:])

    tests = {
        "static"  : StaticPredictor(),
        "bimodal" : BimodalPredictor(n=10),
        "gshare"  : GSharePredictor(n=10),
        "neural"  : NeuralPredictor(traindump)
	#"neural LSTM (ours)"  : NeuralLSTMPredictor(traindump)
    }

    for predictor in tests:
        print("{} predictor had {} accuracy".format(
            predictor, evaluate(tests[predictor], testdump)))
    visualize_test(memdump)
        
if __name__ == "__main__":
    main(filename="data/gcc-1K.trace")
