#!/bin/bash 

export CUDA_VISIBLE_DEVICES="3"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.plot_activations_cifar nn.classifier=Implicit_Divisive_Adaptive_Threshold_VGG nn.threshold=1.0 nn.divisive.sigma=0.1 train.epochs=100 train.reg.l1.scale=0.001 train.reg.hebbian.scale=0.001"
echo $COMMAND
eval $COMMAND


