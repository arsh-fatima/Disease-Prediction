from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

read_pkl=pickle.load(open('diseasepredict.pkl','rb'))
read_pkl2=pickle.load(open('intensitypredict.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/information')
def info():
    return render_template('DiseaseInfo.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/predictDisease')
def predictD():
    return render_template('predictDisease.html')

@app.route('/predictIntensity')
def predictI():
    return render_template('predictIntensity.html')

@app.route('/predictD', methods=['POST'])
def predictDisease():
    float_features=[float(x) for x in request.form.values()]
    final=[np.array(float_features)]
    print(final)
    pred = read_pkl.predict(final)
    return render_template('after.html', data=pred)

@app.route('/predictI', methods=['POST'])
def predictIntensity():

    float_features2=[float(x) for x in request.form.values()]
    final2=[np.array(float_features2)]
    print(final2)
    pred2 = read_pkl2.predict(final2)
    print(pred2)
    return render_template('after2.html', data=pred2)

   
if __name__ == "__main__":
        app.run(debug=True)
