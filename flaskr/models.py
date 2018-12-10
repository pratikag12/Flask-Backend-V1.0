from flaskr import db
from flask_restful import fields
import datetime


class SoundData(db.Model,object):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=False, nullable=False)
    file_uri = db.Column(db.String, unique=False, nullable=True)
    length = db.Column(db.Integer, unique=False, nullable=True)
    date = db.Column(db.DateTime, unique=False, nullable=True)

    def __repr__(self):
        return '<id: {}, name: {}, filename: {}>'.format(self.id, self.name, self.file_uri)

    def get_date(self):
        return self.date

    def export_data(self):
        return {
            'id':self.id,
            'name': self.name,
            'file_uri': self.file_uri,
            'length' : self.length,
            'date' : str(self.date)
        }

    #Place Request data
    def import_metadata(self, request):
        try:
            json_data = request.get_json()
            if 'name' in json_data:
                self.name = json_data['name']  #Name of the file, generated by client
            if 'length' in json_data:
                self.length = json_data['length'] #Convert length to int
            if 'date' in json_data:
                self.date = datetime.strptime(json_data['date'])
            else:
                self.date = datetime.datetime.now() #datetimeobject
        except KeyError as e:
            print "Key not found in metadata"

#Marshal with return
#Sound MetaDeta Return
sound_resource = {
	'id' : fields.Integer,
	'name': fields.String,
	'file_uri':fields.String,
	'length': fields.Integer,
	'date': fields.DateTime
}