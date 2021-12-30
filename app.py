from flask import Flask, render_template
import pandas as pd
import numpy as np
import pickle
from flask import request
app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

car = pd.read_csv("cardata.csv")


@app.route('/')
def home():
    year = sorted(car['Year'].unique(), reverse=True)
    fuel_type = car['Fuel_Type'].unique()
    seller_type = car['Seller_Type'].unique()
    transmission_type = car['Transmission'].unique()
    owner_type = car['Owner'].unique()
    return render_template('index.html', year=year, fuel_type=fuel_type, seller_type=seller_type, transmission_type=transmission_type, owner_type=owner_type)


@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form.get('year'))
    pre_price = float(request.form.get('pre_price'))
    kilo_driven = int(request.form.get('kilo_driven'))
    fuel_type = request.form.get('fuel')
    if fuel_type == "Petrol":
        fuel = 0
    elif fuel_type == "Diesel":
        fuel = 1
    else:
        fuel = 2
    seller_type = request.form.get('seller_type')
    if seller_type == "Dealer":
        seller = 0
    else:
        seller = 1
    transmission_type = request.form.get('transmission_type')
    if transmission_type == "Manual":
        transmission = 0
    else:
        transmission = 1
    owner_type = int(request.form.get('owner_type'))
    final = np.array([year, pre_price, kilo_driven, fuel,
                     seller, transmission, owner_type])
    prediction = model.predict(final.reshape(1, -1))
    output = round(prediction[0], 2)

    return str(output)


if __name__ == "__main__":
    app.run(debug=True)
