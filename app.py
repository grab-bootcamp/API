from flask import Flask,request
from flask_restful import Resource, Api
from tensorflow import keras
import numpy as np
from flask_cors import CORS

COLUMNS = ['FFMC','DMC','DC','ISI','temp','RH','wind','rain']

app = Flask(__name__)
#
CORS(app)
# creating an API object
api = Api(app)

# Load model
model = keras.models.load_model('model.h5', compile=False)

#prediction api call
class predict(Resource):
    def get(self):
        # Get data
        FFMC = request.args.get('FFMC')
        DMC = request.args.get('DMC')
        DC = request.args.get('DC')
        ISI = request.args.get('ISI')
        temp = request.args.get('temp')
        RH = request.args.get('RH')
        wind = request.args.get('wind')
        rain = request.args.get('rain')

        data = np.array([[float(FFMC),float(DMC),float(DC),float(ISI),float(temp),float(RH),float(wind),float(rain)]])
        
        # Predict
        prediction = model.predict(data)
        prediction = float(prediction[0])

        return prediction

#
api.add_resource(predict, '/predict/')

if __name__ == '__main__':
    app.run()