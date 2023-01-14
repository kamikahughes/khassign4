from flask import Flask, render_template, request
import numpy as np
#import pickle
import joblib

app = Flask(__name__)

filename = 'file_bcanc.pkl'

model = joblib.load(filename)

@app.route('/')

def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    area_worst = request.form['area_worst']
    perimeter_worst = request.form['perimeter_worst']
    perimeter_mean = request.form['perimeter_mean']
    radius_worst = request.form['radius_worst']
          
    pred = model.predict(np.array([[area_worst, perimeter_worst, perimeter_mean, radius_worst]], dtype=float))
    print(pred)
    return render_template('index.html', predict=str(pred))

if __name__ == '__main__':
    app.run
