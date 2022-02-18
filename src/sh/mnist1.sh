#!/bin/bash 

export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.train_mnist nn.classifier=LeNet train.reg.active=[none]"
echo $COMMAND
eval $COMMAND


