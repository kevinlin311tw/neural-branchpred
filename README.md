# neural-branchpred

## install
`pip3 install --upgrade tensorflow-gpu==1.5`

`pip3 install keras, m5`

`sudo ldconfig /usr/local/cuda/lib64`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/`

## prepare data

`bash /data/get_1Mdata.sh`

## quick test on minival

`python3 branch-minival.py`

## run on full dataset

`python3 branch.py`

