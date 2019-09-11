## Convolution Neural Network

Implementations of CNNs from scratch using Numpy. 

Note: The purpose of this repository is to understand what goes benath the layers. In no way it should be used for production purposes nor it aims to replace the existing frameworks. 

## What does it have?

**Network Architecture**
1. Convolution 
2. Relu 
3. Max pool 
4. Convolution
5. Relu
6. Maxpool
7. Flatten
8. Dense
9. Relu
10. Dense
11. Relu 
12. Dense
13. Softmax

Weights are initialized using Xavier initialization and Adam optimizer is used for training the network.

## Usage

1. ```pyenv virtualenv 3.6.5 my-virtual-env-3.6.5``` ; create a virtual environment
2. ```pyenv activate my-virtual-env-3.6.5 ``` ; activate the virtual environment
3. ```pip install -r requirements.txt``` ; Install dependencies
4. ```python run.py``` ;

## Credits 
1. Andrew Ng [course] (https://www.coursera.org/learn/convolutional-neural-networks-tensorflow) on Coursera.
2. [Alex net] (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
3. [Adam optimizer] (https://arxiv.org/pdf/1412.6980.pdf) 
