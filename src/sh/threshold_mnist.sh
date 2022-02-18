#!/bin/bash 

export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.train_mnist --multirun nn.thresholding=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 nn.classifier=ThresholdingLeNet"
echo $COMMAND
eval $COMMAND


