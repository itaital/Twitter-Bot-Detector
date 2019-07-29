# ARCII-for-Matching-Natural-Language-Sentences
A simple version of ARC-II model implemented in Keras.
Please reference paper：<a href='https://arxiv.org/abs/1503.03244'>Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

## Our approch
Based on the assumption that big amount of tweets having similar structure can indicate a bot, we suggest a convolutional architecture.
The architectures include:
-Word embedding.
-Convolution and pooling layers.
-Flattening.
-Multi- layer perceptron (MLP) with activation function.
-Post - processing
-Tweet classification.

# Assumption and goal
Assumption: tweets written by the same generator have the same sentence structure.
Main goal: measure the similarity between two tweets.  

# Algorithm
![alt text](https://cdn1.imggmi.com/uploads/2019/7/29/17e9c0776add8a340818b1121eda87e2-full.png)
