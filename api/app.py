import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

loadModel = pickle.load(open('model.pkl','rb')) 

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    
    '''
    For rendering results on HTML GUI
    '''
    
    fl_features = [float(x) for x in request.form.values()]
    features = [np.array(fl_features)]
    prediction = loadModel.predict(features)
    return  render_template("index.html", prediction_text = "LoanStatus is {}".format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    '''
    For direct API calls trought request
    '''
    
    data = request.get_json(force=True)
    prediction = loadModel.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)