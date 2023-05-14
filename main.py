from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/', methods=["GET"])
@app.route('/index', methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    floor = request.form.get('floor')
    distance = request.form.get('distance_from_noise')
    store = request.form.get('no_of_store')
    
    X_features = [np.array([floor, distance, store])]

    X_norm = scaler.transform(X_features)
    prediction = model.predict(X_norm)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted house price is ${output}')