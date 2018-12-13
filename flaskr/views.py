#Create Blueprint
from flask import request, redirect, url_for, abort, jsonify,send_from_directory, Blueprint, json
from flask_restful import Resource, Api
from flaskr.models import SoundData
from flaskr import db
import os
import flaskr.test_rest as tests
from analysis import CNNModel

sound_api_Blueprint = Blueprint('sound_api', __name__)
api = Api(sound_api_Blueprint) #Create Flask Api
db.create_all()

class sound_get_all(Resource):
	def get(self): #Get all sound data files in server
		return jsonify({'SoundDatas': [SoundData.query.get_or_404(Sound.id).export_data() \
										for Sound in SoundData.query.all()]})
	def post(self):
		new_sound_data = SoundData()
		new_sound_data.import_metadata(request) #Pass flask request object to model
		db.session.add(new_sound_data)
		db.session.commit()
		return jsonify(new_sound_data.export_data())

class sound_metadata(Resource):
	def get(self, id): #get single metadata, then return marshalled object
		return jsonify(SoundData.query.get_or_404(id).export_data())

class serve_file(Resource):
	def get(self, filename):
		path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
		return send_from_directory(path, filename)

class delete_sound_file(Resource):
	def get(self, id):
		sounddata = SoundData.query.get_or_404(id)
		#Delete File
		path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
		if(sounddata.file_uri != None):
			if(os.path.exists(os.path.join(path, sounddata.file_uri))):
				os.remove(os.path.join(path, sounddata.file_uri))
			#Remove From database
		db.session.delete(sounddata)
		db.session.commit()

class delete_sound_files(Resource):
	def get(self):
		#Path for uploads
		d1 = delete_sound_file()
		#Delete all Files
		for Sound in SoundData.query.all():
			d1.get(Sound.id)

class sound_file(Resource):
	def get(self,id):
		servefile = serve_file()
		sound_file = SoundData.query.get_or_404(id)
		return servefile.get(sound_file.file_uri) #get the soundfile

	def post(self,id):
		sound_file = SoundData.query.get_or_404(id)
		if sound_file:
			sounddata = request.data #Incoming Data as String
			path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
			filename = 'StethoData_%s_%s.dat' %(request.date, id)
			with open(os.path.join(path, filename), 'wb') as fp:
				fp.write(sounddata) #Write the sound data
			sound_file.file_uri = filename
			db.session.add(sound_file) #Update the SQL Data field for file URI
			db.session.commit()
			return (204) #Return No Content To Return
		abort(404)

#Analysis API's
class analyze_sound_file(Resource):
	def get(self, id):
		result = 2 #Indicates file does not exist
		soundmetadat = SoundData.query.get_or_404(id) #get sound file from database
		if soundmetadat:
			sound_file = soundmetadat.file_uri.replace("dat", "npy")
			path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mldata') #Check if file exist
			if(os.path.exists(os.path.join(path, sound_file))):
				ml = CNNModel()
				result = ml.runanalysis(os.path.join(path, sound_file))
				#Perform analysis
		return jsonify({'result':result})

#Flask-Restful Api add URL Redirect
api.add_resource(sound_get_all, '/api/v1_0/soundmetadatas')
api.add_resource(sound_metadata, '/api/v1_0/soundmetadata/<int:id>')
api.add_resource(sound_file, '/api/v1_0/sounddata/<int:id>')
api.add_resource(serve_file, '/api/v1_0/file/<string:filename>')
api.add_resource(delete_sound_file, '/api/v1_0/delete/<int:id>')
api.add_resource(delete_sound_files, '/api/v1_0/deleteall')
api.add_resource(analyze_sound_file, '/api/v1_0/analyze/<int:id>')

#Test APi
api.add_resource(tests.test_add_metadata,'/api/v1_0/test')