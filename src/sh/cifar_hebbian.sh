#!/bin/bash 

export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.train_hebbian_cifar nn.classifier=Implicit_Divisive_VGG nn.divisive.sigma=0.5 train.epochs=100 train.reg.active=hebbian train.reg.hebbian.tobe_regularized=['features.1','features.4','features.8']"
echo $COMMAND
eval $COMMAND


