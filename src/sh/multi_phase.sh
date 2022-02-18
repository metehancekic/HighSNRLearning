#!/bin/bash 

export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.train_multi_phase_div --multirun train.reg.active=['hebbian','l1_conv1'] nn.implicit_normalization=l2 nn.normalize_input=false train.reg.hebbian.lamda=0.1 train.reg.hebbian.k=5 nn.divisive.sigma=0.5 nn.thresholding=[0.8,0.8] train.reg.l1.scale=0.00001"
echo $COMMAND
eval $COMMAND


