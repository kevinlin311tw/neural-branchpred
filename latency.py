import timeit
from timeit import default_timer as timer

SETUP_CODE = '''
from branch import preprocess
from predictors.static  import StaticPredictor
from predictors.bimodal import BimodalPredictor
from predictors.gshare  import GSharePredictor
import numpy as np
   
filename = "data/gcc-10M.trace"
memdump = preprocess(filename)
#split = np.array_split(memdump, 1000)
#testdump = split[0]
#traindump = np.concatenate(split[1:])
i = int(memdump.shape[0]/1000)
testdump = memdump[:i]
traindump = memdump[i:]

print("TEST DATA shape: {}".format(testdump.shape))

predictor = GSharePredictor(n=8) 
'''

TEST_CODE = '''
for i in range(len(testdump)):
    pred = predictor.predict(testdump[i])
'''
start = timer()
times = timeit.timeit(setup = SETUP_CODE, stmt = TEST_CODE, number=1)
end = timer()

print("time elapsed: {}".format(end - start))
with open("static_latency.log", mode="w") as f:
   f.write("static took time {} to predict".format(times))

