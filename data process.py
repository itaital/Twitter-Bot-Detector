#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:



import pandas as pd
import os as os
import io
import nltk
import string
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import gensim.downloader as api
from sklearn.model_selection import train_test_split


# In[2]:


## Functions


# In[3]:


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[4]:


def prepare_dataset(dataset=None, x_text=None, y=None):
    try:
        os.mkdir("C:/Users/itaital/Desktop/Python/proj implementation/myImplementation/processed_Data")
        #os.mkdir("C:/Users/ARIEL/Desktop/ariel/proj implementation/myImplementation/processed_Data")
    except OSError:
        pass
    filenames = ['train_X', 'train_y', 'test_X', 'test_y']
    files = []
    for filename in filenames:
        files.append(open(r'C:/Users/itaital/Desktop/Python/proj implementation/myImplementation/processed_Data/' + filename, 'wb'))
        #files.append(open(r'C:/Users/ARIEL/Desktop/ariel/proj implementation/myImplementation/processed_Data/' + filename, 'wb'))

    x_text = fd['process_tweets'].tolist()
    y = fd['label'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(x_text,y,test_size=0.30) # here 30% test 70% train we can change it  
    for item in X_train:
        files[0].write((str(item) +"\n").encode('utf-8'))
    for item in X_test:
        files[2].write((str(item) +"\n").encode('utf-8'))
    for item in y_train:
        files[1].write((str(item) +"\n").encode('utf-8'))
    for item in y_test:
        files[3].write((str(item) +"\n").encode('utf-8'))

    for f in files:
        f.close()


# In[84]:


def sentence2vec(data):
    in_path = data
    max_sentence_lenght = 0
    in_file = open(r'C:/Users/itaital/Desktop/Python/proj implementation/myImplementation/processed_Data/' + in_path, 'rb')
    out_file = open(r'C:/Users/itaital/Desktop/Python/proj implementation/myImplementation/processed_Data/' + in_path + '_vectors', 'wb')
    #in_file = open(r'C:/Users/ARIEL/Desktop/ariel/proj implementation/myImplementation/processed_Data/' + in_path, 'rb')
    #out_file = open(r'C:/Users/ARIEL/Desktop/ariel/proj implementation/myImplementation/processed_Data/' + in_path + '_vectors', 'wb')


    lines = in_file.read().decode('utf-8').splitlines()
    for line in lines:
        splitedTweet = line.split(' ')
        sentence2vec = []
        for word in splitedTweet:
            word = word[1:-2] # only in zeheb dataset
            try:
                sentence2vec.append(dataset.wv[word])
            except:
                continue
        out_file.write(str(len(sentence2vec)).encode('utf-8') + b'\n')
        out_file.write(b',\n'.join(str(vector).encode('utf-8') for vector in sentence2vec) + b'.\n')
    return sentence2vec


# ## files handeling

# In[6]:


fd1 = pd.read_fwf('C:/Users/itaital/rt-polarity.txt', header=None)
fd1.columns = ['tweets']
fd1['label'] = 0


# In[7]:


fd2 = pd.read_fwf('C:/Users/itaital/rt-polarity_y.txt', header=None)
fd2 = fd2[[0]]
fd2.columns = ['tweets']
fd2['label']=1


# In[8]:


fd = fd1.append(fd2)


# In[9]:


fd.reset_index(drop = True,inplace = True)


# ## chack the dataset

# In[ ]:


#fd.head()
#fd.tail()
#fd.info()


# In[ ]:


#nltk.download('stopwords')
#only 1 time


# ## text processing
# 1. remove punc
# 2. remove stop words
# 3. return list of clean stop words

# In[11]:


fd['process_tweets'] = fd['tweets']
fd['process_tweets'] = fd['process_tweets'].apply(text_process)


# ## word2vect model
# the model in wv_from_text

# In[12]:


dataset = KeyedVectors.load_word2vec_format(datapath('wiki-news-300d-1M.vec'), binary=False)  # C text format
#dataset = api.load("glove-twitter-25")  --> second option
vectors = dataset.wv
""" check the model!!! """
# result = dataset.most_similar(positive=['woman', 'king'], negative=['man'])
# print("{}: {:.4f}".format(*result[0]))
#sim_words = dataset.wv.most_similar('dog')
#dataset.wv["king"]


# ## Build test and train files
# don't take time - prepare dataset: open train and test X and Y files
# 

# In[13]:


x_text = fd['process_tweets'].tolist()
y = fd['label'].tolist()
prepare_dataset(x_text = x_text, y = y)


# ## Word2vec
# create 2 files of the tweets converted to vectors (word2vec).
# the fornat is:
# first - the lenght of the sentence (only the words that converted to vector counts)
# than - the arrays of the word2vec, each array has 300 values
# get the max sentence lenght, needed for padding

# In[85]:


sentence2vec('train_X')
sentence2vec('test_X')


# ## Convolution and polling missing!
# 
# the most simple use with karas -> https://keras.io/layers/convolutional/,                          note: in there is the default paramater we need to ue them in the start
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

# QUESTION:
#     1. CLEANING IF IT IS OK
#     2. WORD WITH TYPO CONNECTED WORD

# In[86]:


fd.head()


# In[ ]:




