from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# @app.route("/")
# def index():
#     return "Hello from Space! 🚀"

@app.route('/', methods=["GET"])
@app.route('/index', methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    temp = list(request.form.values())
    int_features = [int(x) for x in temp[0].split(' ')]
    print(int_features)
    final_features = [np.array(int_features)]
    data = scaler.transform(final_features)
    prediction = model.predict(data)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted house price is {}'.format(output))