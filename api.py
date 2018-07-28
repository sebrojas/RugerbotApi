from flask import Response, json, request, jsonify, Flask
from flask_restful import reqparse, abort, Api, Resource
import chatbot_framework as cf
import chatbot_trainer as ct
import entities_framework as ef
import entities_trainer as et
import numpy
import json

app = Flask(__name__)
api = Api(app)



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

parser = reqparse.RequestParser()
parser.add_argument('task')

class TrainChatbot(Resource):
    def post(self):
        data = request.get_json(force = True)
        response = jsonify(ct.train_chatbot_model(data['chatbot_id']))
        response.status_code = 200
        return  response
        
class TrainEntities(Resource):
    def post(self):
        data = request.get_json(force = True)
        response = jsonify(et.train_entities_model(data['business_id']))
        response.status_code = 200
        return response

class GetResponse(Resource):
    def post(self):
        data = request.get_json(force = True)
        response = enc.encode(cf.get_response_id(data['message'],data['chatbot_id']))
        return  response

class GetEntities(Resource):
    def post(self):
        data = request.get_json(force = True)
        response = enc.encode(ef.classify(data['message'],data['business_id']))
        return  response



##
## Actually setup the Api resource routing here
##
api.add_resource(GetResponse, '/getResponse')
api.add_resource(GetEntities, '/getEntities')
api.add_resource(TrainChatbot, '/trainChatbot')
api.add_resource(TrainEntities, '/trainEntities')

if __name__ == '__main__':
    app.run(debug=True)

