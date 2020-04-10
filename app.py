from flask import Flask, request, render_template
import pickle
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, static_folder='templates/assets')

@app.route('/')
def home():
	# return render_template('home.html')
	return render_template('triagem.html')

@app.route('/getdelay', methods=['POST','GET'])
def get_delay():
    if request.method=='POST':

        result = request.form

        age = float(result['age'])
        leukocytes = float(result['leukocytes'])
        monocytes = float(result['monocytes'])
        platelets = float(result['platelets'])

        leukocytes = leukocytes / 11e3 * 6 - 2
        monocytes = monocytes / 1e3 * 6 - 2
        platelets = platelets / 450e3 * 6 - 2
        # age = round(age / 20)

        cat_vector = [[leukocytes, monocytes, platelets, age]]
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('logmodel.pkl', 'rb') as f:
            logmodel = pickle.load(f)

        cat_vector = scaler.transform(cat_vector)
        prediction = logmodel.predict(cat_vector)
        
        return render_template('result.html', prediction=prediction)

    
if __name__ == '__main__':
	app.debug = True
	app.run()