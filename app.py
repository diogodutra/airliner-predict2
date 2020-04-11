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

@app.route('/triagem-resultado', methods=['POST','GET'])
def get_delay():
    if request.method=='POST':

        dict_results = {
            0: ('Negativo', 'Dispensar do teste SARS-CoV-2 RT-PCR.', '#b4ecb4', 'icon-check'),
            1: ('Inconclusivo', 'Encaminhar ao teste SARS-CoV-2 RT-PCR.', '#ffd394', 'icon-question'),
        }

        result = request.form

        str_age = result['age']
        str_leukocytes = result['leukocytes']
        str_monocytes = result['monocytes']
        str_platelets = result['platelets']

        age = float(str_age) # / 20 # round()
        leukocytes = float(str_leukocytes) / 11e3 * 6 - 2
        monocytes = float(str_monocytes) / 1e3 * 6 - 2
        platelets = float(str_platelets) / 450e3 * 6 - 2

        cat_vector = [[leukocytes, monocytes, platelets, age]]
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('logmodel.pkl', 'rb') as f:
            logmodel = pickle.load(f)

        cat_vector = scaler.transform(cat_vector)
        prediction = logmodel.predict(cat_vector)
        
        # return render_template('result.html', prediction=prediction)
        return render_template('triagem-resultado.html',
                                result=dict_results[prediction[0]][0],
                                legend=dict_results[prediction[0]][1],
                                bg_color=dict_results[prediction[0]][2],
                                icon=dict_results[prediction[0]][3],
                                age=str_age,
                                leukocytes=str_leukocytes,
                                monocytes=str_monocytes,
                                platelets=str_platelets,
                                )

    
if __name__ == '__main__':
	app.debug = True
	app.run()