#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/icip/src/lib/"




COMMAND="python -m src.plot_neuron_regime train.type=standard nn.classifier=Implicit_Divisive_Adaptive_Threshold_VGG nn.threshold=1.0 nn.divisive.sigma=0.0 nn.lr=0.001 train.epochs=100 train.reg.l1.scale=0.001 train.reg.hebbian.scale=0.456 train.reg.active=['l1_weight','hebbian'] train.reg.hebbian.tobe_regularized=Conv2d"
echo $COMMAND
eval $COMMAND


