#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.correlations_second nn.implicit_normalization=l2 nn.normalize_input=false train.reg.hebbian.lamda=0.1 train.reg.hebbian.k=5 nn.divisive.sigma=0.5 nn.thresholding=0.8"
echo $COMMAND
eval $COMMAND




