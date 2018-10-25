from flask import Response, json, request, jsonify, Flask

import pymongo
from pymongo import MongoClient
import numpy
import json

app = Flask(__name__)
mongo = MongoClient('mongodb://admin:Ruger19901997!@rugercloud.cloudapp.net/?maxIdleTimeMS=60000')

import chatbot_framework as cf
import chatbot_trainer as ct
import entities_framework as ef
import entities_trainer as et

class MultiDimensionalArrayEncoder(json.JSONEncoder):
    def encode(self, obj):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return {hint_tuples(item[0]): hint_tuples(item[1])}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: hint_tuples(value) for key, value in item.items()}
            if isinstance(item, numpy.generic):
                return numpy.asscalar(item)
            else:
                return item

        return hint_tuples(obj)

enc = MultiDimensionalArrayEncoder()

@app.route('/trainchatbot', methods = ['POST'])
def TrainChatbot():
    data = request.get_json(force = True)
    response = jsonify(ct.train_chatbot_model(data['chatbot_id']))
    return  response,200

@app.route('/trainentities', methods = ['POST'])
def TrainEntities():
    data = request.get_json(force = True)
    response = jsonify(et.train_entities_model(data['business_id']))
    return response,200

@app.route('/getentities', methods = ['POST'])
def GetEntities():
    data = request.get_json(force = True)
    response = enc.encode(ef.classify(data['message'],data['business_id']))
    return  jsonify(response),200

@app.route('/getresponse', methods = ['POST'])
def GetResponseNew():
    data = request.get_json(force = True)
    response = enc.encode(cf.get_response_id(data['message'],data['chatbot_id']))
    return jsonify(response),200

if __name__ == '__main__':
    app.run(debug = True)

