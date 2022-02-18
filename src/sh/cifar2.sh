#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/icip/src/lib/"


COMMAND="python -m src.train_cifar --multirun 
                train.type=standard 
                nn.classifier=Matching_VGG
                nn.threshold=1.0 
                nn.divisive.sigma=0.0 
                nn.lr=0.001 
                train.epochs=100 
                train.regularizer.l1_weight.scale=0.001 
                train.regularizer.active=['l1_weight','matching'] 
                train.regularizer.matching.layer=Conv2d
                train.regularizer.matching.alpha=[0.07,0.018,0.0095,0.015,0.01,0.00000001,0.0,0.0,0.0,0.0,0.0,0.0,0.0]"
echo $COMMAND
eval $COMMAND


