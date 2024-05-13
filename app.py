import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from flask.views import View

reg_model = pickle.load(open('regmode.pkl','rb'))
scale_model = pickle.load(open('scaling.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_data():
    data=request.json['data']
    print(data)
    values = list(data.values())
    numpy_array = np.array(values).reshape(1,-1)
    new_data = scale_model.transform(numpy_array)
    output = reg_model.predict(new_data)
    print(output)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scale_model.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=reg_model.predict(final_input)[0]
    return render_template("home.html", predition_data = "The House price prediction is {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)