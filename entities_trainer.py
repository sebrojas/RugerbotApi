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

def train_entities_model(business_id):
	for document in db.Entities.find({"BusinessId": business_id}):
		print(document['Fields'])

	entities = db.Entities.find({"BusinessId": business_id})
	words = []
	classes = []
	documents = []
	ignore_words = ['?']	

	for entity in entities:
	    for field in entity['Fields']:
	        
	        if field["FieldName"] == "Nombre":
	        	w = nltk.word_tokenize(field['FieldData'])

	        	words.extend(w)

	        	documents.append((w, str(entity['_id'])))
	        
	        	if str(entity['_id']) not in classes:
	           		classes.append(str(entity['_id']))

	words = [stemmer.stem(w.lower()) for w in words]
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
	model.save("./models/business_"+business_id+"/entities.model")

	import pickle
	pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "./models/business_"+business_id+"/entities_training_data", "wb" ) )

	return True