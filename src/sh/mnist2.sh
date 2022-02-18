#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.train_mnist_kpp --multirun nn.implicit_normalization=l2 nn.normalize_input=false,true train.reg.hebbian.lamda=0.1,0.5 train.reg.hebbian.k=5,1,10"
echo $COMMAND
eval $COMMAND


