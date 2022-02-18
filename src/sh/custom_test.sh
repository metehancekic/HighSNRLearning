#!/bin/bash 

export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.custom_test nn.implicit_normalization=l2 nn.normalize_input=false train.reg.hebbian.lamda=0.1 train.reg.hebbian.k=5 nn.divisive.sigma=0.5 nn.thresholding=0.8"
echo $COMMAND
eval $COMMAND

# COMMAND="python -m src.attack_mnist --multirun nn.thresholding=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 nn.classifier=ThresholdingLeNet"
# echo $COMMAND
# eval $COMMAND

