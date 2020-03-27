#!/bin/bash



#download MNIST database
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz 
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
#unpack & rename
gunzip train-labels-idx1-ubyte.gz
mv train-labels-idx1-ubyte train_labels
gunzip train-images-idx3-ubyte.gz
mv train-images-idx3-ubyte train_images
gunzip t10k-labels-idx1-ubyte.gz
mv t10k-labels-idx1-ubyte test_labels
gunzip t10k-images-idx3-ubyte.gz 
mv t10k-images-idx3-ubyte test_images




