#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import os as os
import io
import random
import string


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# ## CNN Module

# In[2]:


import os
import numpy as np
import pickle

from keras.activations import relu
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Reshape, Embedding, Flatten, Conv1D, Conv2D, MaxPooling1D ,MaxPooling2D, Activation
from keras.layers import Dropout, concatenate
from keras.utils.vis_utils import model_to_dot

from sklearn.metrics import classification_report

from IPython.display import SVG

from keras.layers.normalization import BatchNormalization

import ManagerModule


# ## Functions

# In[3]:

class UserModule:
    user_tweet = ''
    model_name = 'arcii_first_version_with_two_inputs'
    threshold = 0.5


    #def __init__(user_tweet, model_name = 'arcii_first_version_with_two_inputs', threshold = 0.5):
     #   self.model_name = model_name
      #  self.user_tweet = user_tweet
       # self.threshold = threshold



    def text_process(mess):
        nopunc = [char for char in mess if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        return [word for word in nopunc.split()] #if word.lower() not in stopwords.words('english')]


    # In[10]:


    def predict(self,user_tweet, model_name = 'arcii_first_version_with_two_inputs', threshold = 0.5):

        if(model_name.endswith('.h5') == True):
            model = load_model('Model/' + model_name)
            pic = 'Model/'  + model_name[:-3] + '.pickle'

        else:
            model = load_model('Model/' + model_name + '.h5')
            pic = 'Model/'  + model_name + '.pickle'

        try:
            with open(pic, 'rb') as f:
                bots_list, max_input_lenght, tokenizer = pickle.load(f)
        except Exception as e:
            print(e)
        tweet_to_predict = []
        tweet_to_predict.append(user_tweet)
        mess = tweet_to_predict[0]
        nopunc = [char for char in mess if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        tweet_to_predict[0] =  [word for word in nopunc.split()] #if word.lower() not in stopwords.words('english')]

        sequences_tweet_to_predict = tokenizer.texts_to_sequences(tweet_to_predict)
        padded_tweet_to_predict = pad_sequences(sequences_tweet_to_predict, maxlen=max_input_lenght, padding='post', truncating='post')

        sum = 0
        tweet_to_predict_list = []

        for y in bots_list:
            tweet_to_predict_list.append(padded_tweet_to_predict[0])
        predict_unknown_tweet = model.predict([bots_list,tweet_to_predict_list], verbose=1)
        class_predictions_unknown_tweet = [np.argmax(x) for x in predict_unknown_tweet]
        for z in class_predictions_unknown_tweet:
            sum+=z

        if (sum>=threshold*len(bots_list)):
            return 1
        else:
            return 0
        


    # In[ ]:
    if __name__ == "__main__":
        self.predict(user_tweet, model_name, threshold)






