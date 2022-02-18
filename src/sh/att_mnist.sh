#!/bin/bash 

export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"

# COMMAND="python -m src.attack_mnist --multirun train.reg.active=[hebbian,l1_conv2] train.reg.hebbian.tobe_regularized=['relu2'] nn.implicit_normalization=l2 nn.normalize_input=false train.reg.hebbian.scale=0.1 train.reg.hebbian.lamda=0.1 train.reg.hebbian.k=10 nn.divisive.sigma=0.5 nn.thresholding=[0.8,0.8],[0.8,1.0],[0.8,1.2],[1.0,0.8],[1.0,1.0]"
# echo $COMMAND
# eval $COMMAND

COMMAND="python -m src.attack_cifar nn.classifier=Implicit_VGG"
echo $COMMAND
eval $COMMAND

# COMMAND="python -m src.attack_mnist --multirun nn.thresholding=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 nn.classifier=ThresholdingLeNet"
# echo $COMMAND
# eval $COMMAND

