#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.correlations_kpp nn.classifier=Custom_LeNet"
echo $COMMAND
eval $COMMAND