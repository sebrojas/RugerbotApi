import mongo_connection as conn
import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
from nltk.stem.snowball import SnowballStemmer
from bson.objectid import ObjectId

stemmer = SnowballStemmer("spanish")
cn = conn.context()
db = cn.client.DevelopmentChatbot

def train_chatbot_model(chatbot_id):
	for document in db.Intents.find({"ChatbotAgentId": chatbot_id}):
		print(document['Patterns'])

	intents = db.Intents.find({})
	words = []
	classes = []
	documents = []
	ignore_words = ['?']

	for intent in intents:
	    for pattern in intent['Patterns']:
	        
	        w = nltk.word_tokenize(pattern['PatternData'])
	        
	        words.extend(w)
	        
	        documents.append((w, str(intent['_id'])))
	        
	        if str(intent['_id']) not in classes:
	            classes.append(str(intent['_id']))

	words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
	words = sorted(list(set(words)))

	classes = sorted(list(set(classes)))

	training = []
	output = []

	output_empty = [0] * len(classes)


	for doc in documents:
	    
	    bag = []
	    
	    pattern_words = doc[0]
	    
	    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
	    
	    for w in words:
	        bag.append(1) if w in pattern_words else bag.append(0)

	    
	    output_row = list(output_empty)
	    output_row[classes.index(doc[1])] = 1

	    training.append([bag, output_row])

	random.shuffle(training)
	training = np.array(training)


	train_x = list(training[:,0])
	train_y = list(training[:,1])

	tf.reset_default_graph()

	net = tflearn.input_data(shape=[None, len(train_x[0])])
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
	net = tflearn.regression(net)


	model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

	model.fit(train_x, train_y, n_epoch=1000, batch_size=9, show_metric=True)
	model.save("./models/chatbot_"+chatbot_id+"/chatbot_"+chatbot_id+".model")

	import pickle
	pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "./models/chatbot_"+chatbot_id+"/training_data", "wb" ) )

	return True