import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/predict",methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    ans =  render_template("index.html", prediction_text = prediction)
    temp_dict = {1:'Pass The Test',0:'Fail The Test'}
    prediction = temp_dict[int(prediction[0])]
    # print(prediction)
    # print(prediction[0])
    return render_template("index.html", prediction_text = "Person will {}".format(prediction))
if __name__ == "__main__":
    app.run(debug=True)
Home()

