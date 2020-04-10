from flask import Flask, request, render_template
import pickle
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

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
        
        # with open('scaler.pkl', 'rb') as f:
        #     scaler = pickle.load(f)
        
        with open('logmodel.pkl', 'rb') as f:
            logmodel = pickle.load(f)

        # cat_vector = scaler.transform(cat_vector)
        prediction = logmodel.predict(cat_vector)
        
        return render_template('result.html', prediction=prediction)

    
if __name__ == '__main__':
	app.debug = True
	app.run()