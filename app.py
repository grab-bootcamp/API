from flask import Flask,request
from flask_restful import Resource, Api
from tensorflow import keras
import numpy as np
from flask_cors import CORS

COLUMNS = ['temp', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI', 'RH', 'BUI', 'FWI']

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
        data = np.array([[float(request.args.get(field)) for field in COLUMNS]])
        
        # Predict
        prediction = model.predict(data)
        prediction = float(prediction[0])

        return prediction

#
api.add_resource(predict, '/predict/')

if __name__ == '__main__':
    app.run()