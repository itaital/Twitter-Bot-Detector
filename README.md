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
<blockquote class="imgur-embed-pub" lang="en" data-id="a/Y2dmmmY" data-context="false" ><a href="//imgur.com/a/Y2dmmmY"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
