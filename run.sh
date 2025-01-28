#!/bin/bash  
  
# Navigate to the script directory  
# cd /path/to/your/script  
  
# Execute the Python script with different arguments  
# ./shell_scripts.sh -m main
./shell_scripts.sh -m imagenet
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 199
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 192
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 183
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 175
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 166
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 114
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 179
# [199, 192, 183, 175, 166, 114, 179]
