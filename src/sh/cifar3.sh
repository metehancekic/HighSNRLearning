#!/bin/bash 

export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="/home/metehan/icip/src/lib/"


COMMAND="python -m src.train_cifar --multirun 
                train.type=adversarial 
                nn.classifier=Matching_VGG
                nn.conv_layer_type=conv2d
                nn.threshold=0.0 
                nn.divisive.sigma=None
                nn.lr=0.001 
                train.epochs=100 
                train.regularizer.l1_weight.scale=0.001 
                train.regularizer.active=None
                train.regularizer.matching.layer=Conv2d"
echo $COMMAND
eval $COMMAND


