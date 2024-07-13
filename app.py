from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

## loading the pretrained model 
model = joblib.load('model/heart_disease_model.pkl')

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    ## get data from form
    input_data = request.form.to_dict()
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    ## to make a more descriptive answer 

    if prediction == 1: 
        result = 'Heart Disease'
    else: 
        result = 'No Heart Disease'
    return render_template('predictemp.html', prediction=result)


if __name__ == '__main__': 
    app.run(debug=True)