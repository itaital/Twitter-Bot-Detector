#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[14]:

import keras
import pandas as pd
import os as os
import io
import random
import string
from keras import backend as K
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from datetime import datetime


# ## CNN Module

# In[15]:


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


# In[16]:

class ManagerModule:
	#global variables

	bots_list= None
	max_input_lenght = None
	tokenizer = None


	# ## Functions

	# In[17]:


	def __init__(self, train_perc=0.7, batch_size=50, epoch_nbr=10, model_name = 'arcii_first_version_with_two_inputs', calback_func=None, output=None, log_folder_name=None):
		K.clear_session()
		if log_folder_name is None:
			self.name = datetime.now().strftime('%Y%m%d_%H_%M_%S')
		else:
			self.name = log_folder_name
		self.isRun = False
		self.AccuracyCallback = calback_func
		self.output = output
		self.model_name = model_name
		self.train_percent = train_perc
		self.batch_size = batch_size
		self.epoch_nbr = epoch_nbr
		self.session = tf.Session()
		#self.create_model(model_name, bots_file_path = 'db/bots_tweets.txt' , human_file_path = 'db/human_tweets.txt', test_size_input = 1- train_perc, batch_size=batch_size, epochs=epoch_nbr)
		self.default_graph = tf.get_default_graph()

	def set_running_status(self, isRun):
		"""
		function to control learning process
		:param isRun: change status of thread
		:return:
		"""
		self.isRun = isRun

	def text_process(self,mess):
		nopunc = [char for char in mess if char not in string.punctuation]
		nopunc = ''.join(nopunc)
		return [word for word in nopunc.split()] #if word.lower() not in stopwords.words('english')]


	# ## CNN Utils Functions

	# In[18]:


	def compute_metrics(self,raw_predictions, label_encoder):
		# convert raw predictions to class indexes
		threshold = 0.5
		class_predictions = [(x > threshold).astype(int) for x in model.predict(x_test)]

		# convert raw predictions to class indexes
		threshold = 0.5
		class_predictions = [(x > threshold).qrcype(int) for x in model.predict(x_test)]

		# select only one class (i.e., the dim in the vector with 1.0 all other are at 0.0)
		class_index = ([np.argmax(x) for x in class_predictions])

		# convert back to original class names
		pred_classes = label_encoder.inverse_transform(class_index)

		# print precision, recall, f1-score report
		print(classification_report(y_test, pred_classes))

	def load_fasttext_embeddings(self):
		glove_dir = os.getcwd()
		embeddings_index = {}
		f = open(os.path.join(glove_dir, 'db/wiki-news-300d-1M.vec'), encoding="utf8")
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()
		print('Found %s word vectors.' % len(embeddings_index))
		return embeddings_index

	def create_embeddings_matrix(self,embeddings_index, vocabulary, embedding_dim=100):
		try:
			embeddings_matrix = np.random.rand(len(vocabulary)+1, embedding_dim)
		except Exception as e:
			print(e)
		for i, word in enumerate(vocabulary):
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embeddings_matrix[i] = embedding_vector
		print('Matrix shape: {}'.format(embeddings_matrix.shape))
		return embeddings_matrix


	def get_embeddings_layer(self,embeddings_matrix, name, max_len, trainable=False):
		embedding_layer = Embedding(
			input_dim=embeddings_matrix.shape[0],
			output_dim=embeddings_matrix.shape[1],
			input_length=max_len,
			weights=[embeddings_matrix],
			trainable=trainable,
			name=name)
		return embedding_layer


	def get_conv_pool_arc_II(self,x_input, sufix,maxlen, n_grams=[3,4,5], feature_maps=100): #maybe we need to change feature_maps = 500
		branches = []
		kernel_size_1d = 3
		num_conv2d_layers = 2
		filters_2d=[256,128]
		kernel_size_2d=[[3,3], [3,3]]
		mpool_size_2d=[[2,2], [2,2]]
		dropout_rate=0
		
		layer1_conv = Conv1D(filters=maxlen, kernel_size=kernel_size_1d, padding='same')(x_input)
		layer1_activation=Activation('relu')(layer1_conv)
		layer1_reshaped=Reshape((maxlen, maxlen, -1))(layer1_activation)
		z=MaxPooling2D(pool_size=(2,2))(layer1_reshaped)

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
	def get_cnn_pre_trained_embeddings(self,embedding_layer, max_len):
		# connect the input with the embedding layer
		input_1 = Input(shape=(max_len,), dtype='int32', name='input_1')
		input_2 = Input(shape=(max_len,), dtype='int32', name='input_2')
		x_1 = embedding_layer(input_1)
		x_2 = embedding_layer(input_2)
		
		layer1_input=concatenate([x_1, x_2])

		# generate several branches in the network, each for a different convolution+pooling operation,
		# and concatenate the result of each branch into a single vector
		mlp1 = self.get_conv_pool_arc_II(layer1_input, 'static',max_len)

		# pass the concatenated vector to the predition layer
		o = Dense(2, activation='sigmoid', name='output')(mlp1)

		model = Model(inputs=[input_1, input_2], outputs=o)
		model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam', metrics=['accuracy'])

		return model


	# ## Build test and train files
	# don't take time - prepare dataset: open train and test X and Y files
	# 

	# In[19]:


	def Create_Tokenizer(self,text_for_tokenizer):
		global tokenizer
		tokenizer.fit_on_texts(text_for_tokenizer)


	# In[20]:


	def Build_Train_Set(self,X_train, y_train):
		global bots_list
		global max_input_lenght
		global tokenizer
		
		# built two lists with tweets and labels
		tweets_train = [x for x in X_train]
		labels_train = [y for y in y_train]

		# convert list of tokens/words to indexes
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
		i=0
		for label in labels_train:
			if(label == 0):
				bots_list.append(data_train[i])
			i = i+1

		#train - prepare data set for random tweets
		index =0
		tweets_list = []
		final_label =[]
		while(index < len(bots_list)):
			i = random.sample(range(1, len(data_train)), 1)[0]
			tweets_list.append(data_train[i])
			final_label.append(1 if labels_train[i] == 0 else 0)
			index = index + 1


		# Encode the labels, each must be a vector with dim = num. of possible labels
		le = LabelEncoder()
		le.fit(final_label)
		labels_encoded_train = le.transform(final_label)
		categorical_labels_train = to_categorical(labels_encoded_train, num_classes=None)
		print('Shape of train data tensor:', data_train.shape)
		print('Shape of train label tensor:', categorical_labels_train.shape)
		
		return tweets_list, categorical_labels_train, le, word_index


	# ## test

	# In[21]:


	def Build_Test_Set(self,X_test, y_test, le):
		
		global bots_list
		global max_input_lenght
		global tokenizer
				
		# pre-process test data
		tweets_test = [x for x in X_test]
		labels_test = [y for y in y_test]

		# convert list of tokens/words to indexes
		sequences_test = tokenizer.texts_to_sequences(tweets_test)
		x_test = pad_sequences(sequences_test, maxlen=max_input_lenght)

		#test - prepare data set for random tweets
		index =0
		test_tweets_list = []
		test_final_label =[]
		while(index < len(bots_list)):
			i = random.sample(range(1, len(x_test)), 1)[0]
			test_tweets_list.append(x_test[i])
			test_final_label.append(1 if labels_test[i] == 0 else 0)
			index = index + 1

		le.fit(test_final_label)
		labels_encoded_test = le.transform(test_final_label)
		categorical_labels_test = to_categorical(labels_encoded_test, num_classes=None)
		print('Shape of test data tensor:', len(test_tweets_list))
		print('Shape of test labels tensor:', categorical_labels_test.shape)
		
		return test_tweets_list, categorical_labels_test


	# # important  - the first and the fifth tweet is also bot and human

	# ## CNN with pre-trained static word embeddings

	# In[22]:


	def build_model(self,word_index):
		global max_input_lenght
		
		embeddings_index = self.load_fasttext_embeddings()
		embeddings_matrix = self.create_embeddings_matrix(embeddings_index, word_index, 300)
		embedding_layer_static = self.get_embeddings_layer(embeddings_matrix, 'embedding_layer_static', max_input_lenght, trainable=False)
		model = self.get_cnn_pre_trained_embeddings(embedding_layer_static, max_input_lenght)
		return model


	def getCallBacks(self):
			"""
			callbacks to use when training model.
			- Early stopping to stop training if it;s going to be overfitting and restore best weights.
			- TensorBoard to get option to view logs in dashboard
			- AccuracyCallBack our Callback to print log and redrawing graphs
			"""
			earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss',
													min_delta=.01,
													patience=10,
													verbose=1,
													mode='auto',
													baseline=None,
													restore_best_weights=True)
			tensorBoard = keras.callbacks.TensorBoard(log_dir='./logs/{}/'.format(self.name),
													histogram_freq=0,
													batch_size=32,
													write_graph=True,
													write_grads=True,
													write_images=True,
													embeddings_freq=0,
													embeddings_layer_names=None,
													embeddings_metadata=None,
													embeddings_data=None,
													update_freq='epoch')
			return [tensorBoard, earlyStop, self.AccuracyCallback]

	def create_model(self):
		global bots_list
		global tokenizer
		global max_input_lenght
		bots_file_path = 'db/bots_tweets.txt'
		human_file_path = 'db/human_tweets.txt'
		
		tokenizer = Tokenizer()
		bots_list =[]
		max_input_lenght = 0

		fd1 = pd.read_fwf(bots_file_path, header=None)
		fd1.columns = ['tweets']
		fd1['label'] = 0
		# 0-> bots
		
		fd2 = pd.read_fwf(human_file_path, header=None)
		fd2 = fd2[[0]]
		fd2.columns = ['tweets']
		fd2['label']=1
		# 1-> humans
		
		fd = fd1.append(fd2)
		fd.reset_index(drop = True,inplace = True)
		
		fd['process_tweets'] = fd['tweets']
		fd['process_tweets'] = fd['process_tweets'].apply(self.text_process)
		x_text = fd['process_tweets'].tolist()
		
		#split dataset
		y = fd['label'].tolist()
		test_size_input = 1 - self.train_percent
		X_train, X_test, y_train, y_test = train_test_split(x_text,y,test_size=test_size_input) # here 30% test 70% train we can change it  
		
		print("Train Samples: {}".format(len(X_train)))
		print("Test Samples : {}".format(len(X_test)))
		print("Labels       : {}".format({x for x in y_train}))
		
		#tokenizer = self.Create_Tokenizer(x_text)
		self.Create_Tokenizer(x_text)

		tweets_list, categorical_labels_train, le, word_index = self.Build_Train_Set(X_train, y_train)
		
		test_tweets_list, categorical_labels_test = self.Build_Test_Set(X_test, y_test, le)
		
		model = self.build_model(word_index)
		with open(os.path.join(os.getcwd(), 'Model') + '\\' + self.model_name + '.pickle', 'wb') as f:
			pickle.dump([bots_list, max_input_lenght, tokenizer], f)
		
		history = model.fit([bots_list,tweets_list],
								categorical_labels_train,
								batch_size=self.batch_size,
								epochs= self.epoch_nbr,
								#validation_data=([bots_list,tweets_list], categorical_labels_train),
								callbacks=self.getCallBacks())
		
		model.save(os.path.join(os.getcwd(), 'Model') + '\\' + self.model_name + '.h5')
        

