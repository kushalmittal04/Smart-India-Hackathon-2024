# flask, pandas, scikit-learn, pickle-mixin
from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)
cars = pd.read_csv("cleaned_cars_data.csv")
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

@app.route('/')
def index():
    companies = sorted(cars['company'].unique())
    car_models = sorted(cars['name'].unique())
    years = sorted(cars['year'].unique(), reverse=True)
    fuel_type = sorted(cars['fuel_type'].unique())
    companies.insert(0,"Select Company")
    years.insert(0,"Select Year")
    fuel_type.insert(0,"Select Fuel Type")

    return render_template('index.html',companies=companies, car_models=car_models, years=years, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    # form data extraction
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    # print(company, car_model, year, fuel_type,kms_driven)

    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    #print(prediction[0])
    return str(np.round(prediction[0],2))

if __name__ == "__main__":
    app.run(debug=True)
