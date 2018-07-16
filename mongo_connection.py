import pymongo
from pymongo import MongoClient

class context:
	def __init__(self):
		self.client = MongoClient('mongodb://admin:Ruger19901997!@rugercloud.cloudapp.net/?maxIdleTimeMS=60000')


