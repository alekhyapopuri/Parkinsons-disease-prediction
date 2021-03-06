# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:32:24 2020

@author: ALEKHYA
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
   
    features_name = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
       'spread1', 'spread2', 'D2', 'PPE']
   
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    print(output)
       
    if output == 0:
        res_val = "no parkinsons disease "
    else:
        res_val = "** parkinsons disease **"
       

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))





if __name__ == "__main__":
    app.run(debug=True)
