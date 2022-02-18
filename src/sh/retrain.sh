#!/bin/bash 

export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.retrain_after_thresholding --multirun nn.implicit_normalization=l2 nn.normalize_input=false train.reg.hebbian.lamda=0.1 train.reg.hebbian.k=5 nn.divisive.sigma=0.1,0.5 nn.thresholding=0.8,1.0,1.2,1.4,1.6"
echo $COMMAND
eval $COMMAND


