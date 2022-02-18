#!/bin/bash 

export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.plot_histograms --multirun nn.implicit_normalization=l2 nn.normalize_input=false,true train.reg.hebbian.lamda=0.1 train.reg.hebbian.k=5 nn.divisive.sigma=0.1,0.2,0.3,0.5"
echo $COMMAND
eval $COMMAND


