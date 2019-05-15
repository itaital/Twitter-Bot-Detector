#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import os as os
import io
import random
import nltk
import string
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import gensim.downloader as api

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

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from keras.activations import relu
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Reshape, Embedding, Flatten, Conv1D, Conv2D, MaxPooling1D ,MaxPooling2D, Activation
from keras.layers import Dropout, concatenate
from keras.utils.vis_utils import model_to_dot

from sklearn.metrics import classification_report

from IPython.display import SVG

from keras.layers.normalization import BatchNormalization


# ## Functions

# In[3]:


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split()] #if word.lower() not in stopwords.words('english')]


# ## CNN Utils Functions

# In[4]:


def compute_metrics(raw_predictions, label_encoder):
    # convert raw predictions to class indexes
    threshold = 0.5
    class_predictions = [(x > threshold).astype(int) for x in model.predict(x_test)]

    # convert raw predictions to class indexes
    threshold = 0.5
    class_predictions = [(x > threshold).astype(int) for x in model.predict(x_test)]

    # select only one class (i.e., the dim in the vector with 1.0 all other are at 0.0)
    class_index = ([np.argmax(x) for x in class_predictions])

    # convert back to original class names
    pred_classes = label_encoder.inverse_transform(class_index)

    # print precision, recall, f1-score report
    print(classification_report(y_test, pred_classes))

def load_fasttext_embeddings():
    glove_dir = 'C:/Users/itaital/AppData/Local/Continuum/anaconda3/envs/tensorflow_env/Lib/site-packages/gensim/test/test_data'
    #glove_dir = 'C:/Users/itaital/Desktop/Python/proj implementation/myImplementation'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'wiki-news-300d-1M.vec'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def create_embeddings_matrix(embeddings_index, vocabulary, embedding_dim=100):
    embeddings_matrix = np.random.rand(len(vocabulary)+1, embedding_dim)
    for i, word in enumerate(vocabulary):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    print('Matrix shape: {}'.format(embeddings_matrix.shape))
    return embeddings_matrix


def get_embeddings_layer(embeddings_matrix, name, max_len, trainable=False):
    embedding_layer = Embedding(
        input_dim=embeddings_matrix.shape[0],
        output_dim=embeddings_matrix.shape[1],
        input_length=max_len,
        weights=[embeddings_matrix],
        trainable=trainable,
        name=name)
    return embedding_layer


def get_conv_pool_arc_II(x_input, sufix,maxlen, n_grams=[3,4,5], feature_maps=100): #maybe we need to change feature_maps = 500
    branches = []
    kernel_size_1d = 3
    num_conv2d_layers = 2
    filters_2d=[256,128]
    kernel_size_2d=[[3,3], [3,3]]
    mpool_size_2d=[[2,2], [2,2]]
    dropout_rate=0
    
    layer1_conv = Conv1D(filters=maxlen, kernel_size=kernel_size_1d, padding='same')(x_input)
    layer1_activation=Activation('relu')(layer1_conv)
    print("layer1_activation:")
    print(layer1_activation.shape)
    layer1_reshaped=Reshape((maxlen, maxlen, -1))(layer1_activation)
    z=MaxPooling2D(pool_size=(2,2))(layer1_reshaped)
    #z=MaxPooling2D(pool_size=(2,2))(layer1_activation)

    for i in range(num_conv2d_layers):
        z=Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same')(z)
        z=Activation('relu')(z)
        z=MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z)
    
    pool1_flat=Flatten()(z)
    pool1_flat_drop=Dropout(rate=dropout_rate)(pool1_flat)
    pool1_norm=BatchNormalization()(pool1_flat_drop)
    mlp1=Dense(64)(pool1_norm)
    mlp1=Activation('relu')(mlp1)

    return mlp1
def get_cnn_pre_trained_embeddings(embedding_layer, max_len):
    # connect the input with the embedding layer
    input_1 = Input(shape=(max_len,), dtype='int32', name='input_1')
    input_2 = Input(shape=(max_len,), dtype='int32', name='input_2')
    x_1 = embedding_layer(input_1)
    x_2 = embedding_layer(input_2)
    
    layer1_input=concatenate([x_1, x_2])

    # generate several branches in the network, each for a different convolution+pooling operation,
    # and concatenate the result of each branch into a single vector
    mlp1 = get_conv_pool_arc_II(layer1_input, 'static',max_len)

    # pass the concatenated vector to the predition layer
    o = Dense(1, activation='sigmoid', name='output')(mlp1)

    model = Model(inputs=[input_1, input_2], outputs=o)
    model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam')

    return model


# ## files handeling

# In[5]:


fd1 = pd.read_fwf('C:/Users/itaital/rt-polarity.txt', header=None)
#fd1 = pd.read_fwf('D:/project_files/rt-polarity.txt', header=None)
fd1.columns = ['tweets']
fd1['label'] = 0
# 0-> bots


# In[6]:


fd2 = pd.read_fwf('C:/Users/itaital/rt-polarity_y.txt', header=None)
#fd2 = pd.read_fwf('D:/project_files/rt-polarity_y.txt', header=None)
fd2 = fd2[[0]]
fd2.columns = ['tweets']
fd2['label']=1
# 1-> humans


# In[7]:


fd = fd1.append(fd2)


# In[8]:


fd.reset_index(drop = True,inplace = True)


# ## text processing
# 1. remove punc
# 2. remove stop words
# 3. return list of clean stop words

# In[9]:


fd['process_tweets'] = fd['tweets']
fd['process_tweets'] = fd['process_tweets'].apply(text_process)


# ## Build test and train files
# don't take time - prepare dataset: open train and test X and Y files
# 

# In[54]:


x_text = fd['tweets'].tolist()
y = fd['label'].tolist()
X_train, X_test, y_train, y_test = train_test_split(x_text,y,test_size=0.30) # here 30% test 70% train we can change it  


# In[55]:


print("Train Samples: {}".format(len(X_train)))
print("Test Samples : {}".format(len(X_test)))
print("Labels       : {}".format({x for x in y_train}))


# ## Train

# In[56]:


# built two lists with tweets and labels
tweets_train = [x for x in X_train]
labels_train = [y for y in y_train]

# convert list of tokens/words to indexes
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_train)
sequences_train = tokenizer.texts_to_sequences(tweets_train)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# get the max sentence lenght, needed for padding
max_input_lenght = max([len(x) for x in sequences_train])
print("Max. sequence lenght: ", max_input_lenght)

# pad all the sequences of indexes to the 'max_input_lenght'
data_train = pad_sequences(sequences_train, maxlen=max_input_lenght, padding='post', truncating='post')

#prepare data set of bots
#bots_list = [x for i,x in enumerate(data_train) if labels_train[i] == 0]
bots_list_train = [] # list of all the label of bots.
i=0
for label in labels_train:
    if(label == 0): 
        bots_list_train.append(data_train[i])
    i = i+1

#prepare data set for random tweets
index =0
tweets_list_train = []
final_label_train =[]
while(index < len(bots_list_train)): 
    i = random.sample(range(1, len(bots_list_train)), 1)[0] # random number between 1- size of bots
    tweets_list_train.append(data_train[i])
    final_label_train.append(1 if labels_train[i] == 0 else 0)
    index = index + 1
    

# Encode the labels, each must be a vector with dim = num. of possible labels
le = LabelEncoder()
le.fit(final_label_train)
labels_encoded_train = le.transform(final_label_train)
categorical_labels_train = to_categorical(labels_encoded_train, num_classes=None)
print('Shape of train data tensor:', data_train.shape)
print('Shape of train label tensor:', categorical_labels_train.shape)


# ## Test

# In[57]:


# pre-process test data
tweets_test = [x for x in X_test]
labels_test = [y for y in y_test]
sequences_test = tokenizer.texts_to_sequences(tweets_test)
x_test = pad_sequences(sequences_test, maxlen=max_input_lenght)

bots_list_test = [] # list of all the label of bots.
i=0
for label in labels_test:
    if(label == 0): 
        bots_list_test.append(x_test[i])
    i = i+1

#prepare data set for random tweets
index =0
tweets_list_test = []
final_label_test =[]
while(index < len(bots_list_test)): 
    i = random.sample(range(1, len(bots_list_test)), 1)[0] # random number between 1- size of bots
    tweets_list_test.append(x_test[i])
    final_label_test.append(1 if labels_test[i] == 0 else 0)
    index = index + 1

labels_encoded_test = le.transform(final_label_test)
categorical_labels_test = to_categorical(labels_encoded_test, num_classes=None)
print('Shape of test data tensor:', x_test.shape)
print('Shape of test labels tensor:', categorical_labels_test.shape)


# ## CNN with pre-trained static word embeddings

# In[58]:


embeddings_index = load_fasttext_embeddings()
embeddings_matrix = create_embeddings_matrix(embeddings_index, word_index, 300)
embedding_layer_static = get_embeddings_layer(embeddings_matrix, 'embedding_layer_static', max_input_lenght, trainable=False)
model_2 = get_cnn_pre_trained_embeddings(embedding_layer_static, max_input_lenght)


# ## Train the model

# In[59]:


from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
model_path='C:/Users/itaital/Desktop/Python/proj implementation/myImplementation/arcii_512_128.h5'
log_path='C:/Users\itaital/Desktop/Python/proj implementation/myImplementation/log_arcii_512_128.txt'

batch_size = 128

# build dataset generator
def generator(texts1, texts2, labels, batch_size, min_index, max_index):
    i=min_index
    
    while True:
        if i+batch_size>=max_index:
            i=min_index
        rows=np.arange(i, min(i+batch_size, max_index))
        i+=batch_size
        
        samples1=np.array(texts1)[rows]
        samples2=np.array(texts2)[rows]
        targets=np.array(labels)[rows]
        yield {'input_1':samples1, 'input_2':samples2}, targets



train_gen=generator(bots_list_train, tweets_list_train, final_label_train, batch_size=batch_size, min_index=0, max_index=len(tweets_list_train))

history=model_2.fit_generator(train_gen, epochs=8, steps_per_epoch=len(tweets_list_train)//batch_size,
                  callbacks=[ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True), 
                             EarlyStopping(monitor='val_loss', patience=3), CSVLogger(log_path)])


# ## Predict the classification of the test tweets with the model
# and then compare it to the real classification

# In[62]:


model_2.save('C:/Users/itaital/Desktop/Python/proj implementation/myImplementation/arciiV2.h5')

def test_generator(texts1, texts2, batch_size, min_index, max_index):
    i=min_index
    
    while True:
        if i+batch_size>=max_index:
            i=min_index
        rows=np.arange(i, min(i+batch_size, max_index))
        i+=batch_size
        
        samples1=np.array(texts1)[rows]
        samples2=np.array(texts2)[rows]
        yield {'input_1':samples1, 'input_2':samples2}

test_gen=test_generator(bots_list_test, tweets_list_test, batch_size=1, min_index=0, max_index=len(bots_list_test))
print('Predict...')
#model=load_model(model_path)
#preds=model.predict_generator(test_gen, steps=len(bots_list_test))


# In[ ]:


preds=model_2.predict_generator(test_gen, steps=len(bots_list_test))


# In[ ]:


from sklearn.metrics import confusion_matrix
class_predictions = [np.argmax(x) for x in preds]
print(confusion_matrix(class_predictions, final_label_test))


# In[53]:



#model = load_model('checkpoints/arcii.h5')

