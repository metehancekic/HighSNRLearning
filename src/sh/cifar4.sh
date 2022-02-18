#!/bin/bash 

export CUDA_VISIBLE_DEVICES="3"
export PYTHONPATH="/home/metehan/icip/src/lib/"


COMMAND="python -m src.train_cifar train.type=standard nn.classifier=Standard_Nobias_VGG nn.lr=0.001 train.epochs=100 train.reg.active=none train.reg.hebbian.tobe_regularized=Conv2d"
echo $COMMAND
eval $COMMAND


