from flask import Response, json, request, jsonify, Flask
from flask_restful import reqparse, abort, Api, Resource
import chatbot_framework as cf
import chatbot_trainer as ct
import numpy
import json

app = Flask(__name__)
api = Api(app)



class MultiDimensionalArrayEncoder(json.JSONEncoder):
    def encode(self, obj):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': [hint_tuples(e) for e in item]}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: hint_tuples(value) for key, value in item.items()}
            if isinstance(item, numpy.generic):
                return numpy.asscalar(item)
            else:
                return item

        return super(MultiDimensionalArrayEncoder, self).encode(hint_tuples(obj))

enc = MultiDimensionalArrayEncoder()

parser = reqparse.RequestParser()
parser.add_argument('task')

class TrainChatbot(Resource):
    def post(self):
        data = request.get_json(force = True)
        response = jsonify(ct.train_chatbot_model(data['chatbot_id']))
        response.status_code = 200
        return  response
        

class GetResponse(Resource):
    def post(self):
        data = request.get_json(force = True)
        response = jsonify( enc.encode(cf.get_response_id(data['message'],data['chatbot_id'])) )
        response.status_code = 200
        return  response



##
## Actually setup the Api resource routing here
##
api.add_resource(GetResponse, '/getResponse')
api.add_resource(TrainChatbot, '/trainChatbot')

if __name__ == '__main__':
    app.run(debug=True)

