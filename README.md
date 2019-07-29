# ARCII-for-Matching-Natural-Language-Sentences
A simple version of ARC-II model implemented in Keras.
Please reference paper：<a href='https://arxiv.org/abs/1503.03244'>Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

# Our approch
Based on the assumption that big amount of tweets having similar structure can indicate a bot, we suggest a convolutional architecture.
The architectures include: Word embedding, Convolution and pooling layers, Flattening, Multi- layer perceptron (MLP) with activation function, Post - processing, Tweet classification.

# Assumption and goal
Assumption: tweets written by the same generator have the same sentence structure.
Main goal: measure the similarity between two tweets.  

# Algorithm
![Screenshot](https://i.imgur.com/1yHCGO5.png?raw=true)

We have the tweet the user want to classify. we created a list that contains bot tweets from the dataset at the training phase. Then using our CNN architecture we get the measure of similarity between every bot tweet from the list and the tweet we want to classify. 
One feed forward and back propagate.
Than we convert the degrees to classes when zero means the two tweet have different sentence structure and one means the opposite. 
The next step we sum the classes IDs.
If the percentage of similarity pass the threshold selected by the user, the model will identify the tweet as bot tweet.
Else, we will say that the tweet probably was written by human.

# Result
![Screenshot](https://i.imgur.com/P0EQGDD.png?raw=true)

different threshold value could contribute to various purposes of the application.
Therefore we gave the user the option to change the threshold value according to the purpose for which he use the application.

## How to run the project 
* first things you need to download the DB files and the vector files and put them in the db folder:
1) DB files -
https://drive.google.com/file/d/1zzdaJkl-647LKOXLZ_OUtqQx5Cto7wfJ/view?usp=sharing
https://drive.google.com/file/d/1-8Lz5TLU1uVAvDpjpVbHqGw3OK0sG1Ep/view?usp=sharing
2) vector file - 
https://www.kaggle.com/yesbutwhatdoesitmean/wikinews300d1mvec

* Executable file
If you just want to run the project without a compiler or the liabery.
you can download the Executable file from here:
https://drive.google.com/file/d/1Go1kh0UDIt5eTSDCCWZcYAMuJm_Wavna/view?usp=sharing

* Notbook (There are explanations inside the folder)
in the folder 'Notebook' there is our notbook that have only pure code without GUI.

* Complete project (There are explanations inside the folder).
all the project files are in the 'TwitterBotDetector' folder

* to run the Notbook or the Complete project some liaberys are needed: 


