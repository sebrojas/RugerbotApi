import os
import pickle
import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import codecs
import json
import chatbot_trainer as ct
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("spanish")


def load_model(business_id):
	if(not os.path.isdir("./models/business_"+business_id+"/")):
		ct.train_chatbot_model(chatbot_id)

	data = pickle.load( open( "./models/business_"+business_id+"/entities_training_data", "rb" ) )
	words = data['words']
	classes = data['classes']
	train_x = data['train_x']
	train_y = data['train_y']

	tf.reset_default_graph()

	net = tflearn.input_data(shape=[None, len(train_x[0])])
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
	net = tflearn.regression(net)

	model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
	model.load("./models/business_"+business_id+"/entities.model")

	return model,words,classes

def clean_up_sentence(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	
	sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    
	return sentence_words


def bow(sentence, words, show_details=False):
	sentence_words = clean_up_sentence(sentence)
	bag = [0]*len(words)  
	for s in sentence_words:
		for i,w in enumerate(words):
			if w == s: 
				bag[i] = 1
				if show_details:
					print ("found in bag: %s" % w)
	return(np.array(bag))

context = {}

ERROR_THRESHOLD = 0.25

def classify(sentence,business_id):
	model,words,classes = load_model(business_id)
	results = model.predict([bow(sentence, words)])[0]
	results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append((classes[r[0]], r[1]))
	return return_list
