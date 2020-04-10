from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/getdelay', methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result = request.form
        age = float(result['age'])
        leukocytes = float(result['leukocytes'])
        monocytes = float(result['monocytes'])
        platelets = float(result['platelets'])

        cat_vector = [[leukocytes, monocytes, platelets, age]]
        
        pkl_scaler = open('scaler.pkl', 'rb')
        pkl_model = open('logmodel.pkl', 'rb')

        scaler = pickle.load(pkl_scaler)
        logmodel = pickle.load(pkl_model)

        cat_vector = scaler.transform(cat_vector)
        prediction = logmodel.predict(cat_vector)
        
        return render_template('result.html', prediction=prediction)

    
if __name__ == '__main__':
	app.debug = True
	app.run()