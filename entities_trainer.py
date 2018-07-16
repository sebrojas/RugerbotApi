import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import codecs
import json
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer

Usuarios = {456:{
    'Nombre': '',
    'Email': '',
    'Telefono': '',
}}
entities = {'productos':
                [
                    {
                        'id':1,
                        'string':['galaxy s8'],
                        'precio':4000000
                    },
                    {
                        'id':2,
                        'string':['iphone x'],
                        'precio':7500000
                    },
                    {
                        'id':3,
                        'string':['huawei p20'],
                        'precio':5000000
                    }
                ]
           }

stemmer = SnowballStemmer("spanish")

words = []
classes = []
documents = []
ignore_words = ['?']

for entity in entities['productos']:
    for pattern in entity['string']:
        #print(pattern)
        w = nltk.word_tokenize(pattern)
        #print(w)
        words.extend(w)
        
        documents.append((w, entity['id']))
        
        if entity['id'] not in classes:
            classes.append(entity['id'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))


classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

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

train_x_np = np.array(training[:,0])
train_y_np = np.array(training[:,1])

print (train_x)
#print ("#################################")
print (train_y)

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)


model = tflearn.DNN(net, tensorboard_dir='p_tflearn_logs')

model.fit(train_x, train_y, n_epoch=800, batch_size=9, show_metric=True)
model.save('products_model.tflearn')

import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data_products", "wb" ) )
