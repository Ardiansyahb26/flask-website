import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import logging

app = Flask(__name__)
choicemodel = None
choicemodelsoh = None
battery_type = '18650'
soh = 0
rul = 0

# Konfigurasi Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_models(battery_type):
    global model_rul_lstm, scaler_rul_lstm, model_rul_dtree, scaler_rul_dtree, model_rul_svr, scaler_rul_svr
    global model_soh_lstm, scaler_soh_lstm, model_soh_dtree, scaler_soh_dtree, model_soh_svr, scaler_soh_svr
    global scaler_y_rul, scaler_y_soh, scaler_y_rul_temperatur, scaler_y_soh_temperatur
    global temperature_model_rul_lstm, temperature_scaler_rul_lstm, temperature_model_rul_dtree, temperature_scaler_rul_dtree
    global temperature_model_rul_svr, temperature_scaler_rul_svr, temperature_model_soh_lstm, temperature_scaler_soh_lstm
    global temperature_model_soh_dtree, temperature_scaler_soh_dtree, temperature_model_soh_svr, temperature_scaler_soh_svr

    try:
        # Memuat model RUL
        model_rul_lstm = joblib.load(f'ModelRUL/{battery_type}/lstm/lstm_model.pkl')
        scaler_rul_lstm = joblib.load(f'ModelRUL/{battery_type}/lstm/lstm_scaler.pkl')
        model_rul_dtree = joblib.load(f'ModelRUL/{battery_type}/dtree/dtree.pkl')
        scaler_rul_dtree = joblib.load(f'ModelRUL/{battery_type}/dtree/scalerdtree.pkl')
        model_rul_svr = joblib.load(f'ModelRUL/{battery_type}/svr/svr.pkl')
        scaler_rul_svr = joblib.load(f'ModelRUL/{battery_type}/svr/scalersvr.pkl')

        # Memuat model SOH
        model_soh_lstm = joblib.load(f'ModelSOH/{battery_type}/lstm.pkl')
        scaler_soh_lstm = joblib.load(f'ModelSOH/{battery_type}/lstmscaler.pkl')
        model_soh_dtree = joblib.load(f'ModelSOH/{battery_type}/dtree.pkl')
        scaler_soh_dtree = joblib.load(f'ModelSOH/{battery_type}/scalerdtree.pkl')
        model_soh_svr = joblib.load(f'ModelSOH/{battery_type}/svr.pkl')
        scaler_soh_svr = joblib.load(f'ModelSOH/{battery_type}/scalersvr.pkl')

        # Scaler untuk RUL dan SOH
        scaler_y_rul = MinMaxScaler(feature_range=(0, 100))
        scaler_y_rul.fit(np.array([0, 100]).reshape(-1, 1))
        scaler_y_soh = MinMaxScaler(feature_range=(0, 100))
        scaler_y_soh.fit(np.array([0, 100]).reshape(-1, 1))

        # Memuat model temperatur
        temperature_model_rul_lstm = joblib.load(f'ModelRUL/ModelTemperature/lstm/lstm_model.pkl')
        temperature_scaler_rul_lstm = joblib.load(f'ModelRUL/ModelTemperature/lstm/lstm_scaler.pkl')
        temperature_model_rul_dtree = joblib.load(f'ModelRUL/ModelTemperature/dtree/dtree.pkl')
        temperature_scaler_rul_dtree = joblib.load(f'ModelRUL/ModelTemperature/dtree/scalerdtree.pkl')
        temperature_model_rul_svr = joblib.load(f'ModelRUL/ModelTemperature/svr/svr.pkl')
        temperature_scaler_rul_svr = joblib.load(f'ModelRUL/ModelTemperature/svr/scalersvr.pkl')

        temperature_model_soh_lstm = joblib.load(f'ModelSOH/ModelTemperature/lstm/lstm_model.pkl')
        temperature_scaler_soh_lstm = joblib.load(f'ModelSOH/ModelTemperature/lstm/lstm_scaler.pkl')
        temperature_model_soh_dtree = joblib.load(f'ModelSOH/ModelTemperature/dtree/dtree.pkl')
        temperature_scaler_soh_dtree = joblib.load(f'ModelSOH/ModelTemperature/dtree/scalerdtree.pkl')
        temperature_model_soh_svr = joblib.load(f'ModelSOH/ModelTemperature/svr/svr.pkl')
        temperature_scaler_soh_svr = joblib.load(f'ModelSOH/ModelTemperature/svr/scalersvr.pkl')

        scaler_y_temperatur = MinMaxScaler(feature_range=(0, 100))
        scaler_y_temperatur.fit(np.array([0, 100]).reshape(-1, 1))
        scaler_y_soh_temperatur = MinMaxScaler(feature_range=(0, 100))
        scaler_y_soh_temperatur.fit(np.array([0, 100]).reshape(-1, 1))

    except FileNotFoundError as e:
        logging.error(f"Error loading model: {e}")
    except Exception as e:
        logging.error(f"Unexpected error loading model: {e}")

load_models(battery_type)

def preprocess_input_data_soh_temperatur(soh, model_type):
    X_manual = np.array([[soh]])
    if model_type == 'lstm':
        X_manual_scaled = scaler_soh_lstm.transform(X_manual)
        return X_manual_scaled.reshape((X_manual_scaled.shape[0], 1, X_manual_scaled.shape[1]))
    elif model_type == 'dtree':
        return scaler_soh_dtree.transform(X_manual)
    elif model_type == 'svr':
        return scaler_soh_svr.transform(X_manual)

def predict_soh_temperature(soh):
    try:
        logging.debug(f"SOH: {soh}")
        X_manual_reshaped = preprocess_input_data_soh_temperatur(soh, choicemodelsoh)
        
        if choicemodelsoh == 'lstm':
            soh_prediction_scaled = temperature_model_soh_lstm.predict(X_manual_reshaped)
        elif choicemodelsoh == 'dtree':
            soh_prediction_scaled = temperature_model_soh_dtree.predict(X_manual_reshaped)
        elif choicemodelsoh == 'svr':
            soh_prediction_scaled = temperature_model_soh_svr.predict(X_manual_reshaped)
        else:
            raise ValueError(f"Invalid model choice for SOH temperature prediction: {choicemodelsoh}")
        
        soh_prediction = scaler_y_soh_temperatur.inverse_transform(soh_prediction_scaled.reshape(-1, 1)).flatten()[0]
        return str(float(soh_prediction))[:4]

    except Exception as e:
        logging.error(f"Error in predict_soh_temperature: {e}") 
        return "Error"

def preprocess_input_data_rul(soh, voltage, model_type):
    X_manual = np.array([[soh, voltage]])
    if model_type == 'lstm':
        X_manual_scaled = scaler_rul_lstm.transform(X_manual)
        return X_manual_scaled.reshape((X_manual_scaled.shape[0], 1, X_manual_scaled.shape[1]))
    elif model_type == 'dtree':
        return scaler_rul_dtree.transform(X_manual)
    elif model_type == 'svr':
        return scaler_rul_svr.transform(X_manual)
    
def preprocess_input_data_soh(voltage, model_type):
    X_manual = np.array([[voltage]])
    if model_type == 'lstm':
        X_manual_scaled = scaler_soh_lstm.transform(X_manual)
        return X_manual_scaled.reshape((X_manual_scaled.shape[0], 1, X_manual_scaled.shape[1]))
    elif model_type == 'dtree':
        return scaler_soh_dtree.transform(X_manual)
    elif model_type == 'svr':
        return scaler_soh_svr.transform(X_manual)

def predict_rul(soh, voltage):
    try:
        X_manual_reshaped = preprocess_input_data_rul(soh, voltage, choicemodel)
        if choicemodel == 'lstm':
            rul_prediction_scaled = model_rul_lstm.predict(X_manual_reshaped)
        elif choicemodel == 'dtree':
            rul_prediction_scaled = model_rul_dtree.predict(X_manual_reshaped)
        elif choicemodel == 'svr':
            rul_prediction_scaled = model_rul_svr.predict(X_manual_reshaped)
        else:
            raise ValueError(f"Invalid model choice for RUL prediction: {choicemodel}")

        rul_prediction = scaler_y_rul.inverse_transform(rul_prediction_scaled.reshape(-1, 1)).flatten()[0]
        rul_prediction_int = max(0, int(rul_prediction))

        # Menentukan kapasitas baterai
        kapasitas_nominal = 4973 if battery_type == '21700' else 2955
        kapasitas_mAh = (soh / 100) * kapasitas_nominal

        # Menambahkan kondisi baterai
        if battery_type == '21700':
            if 4.9 <= voltage <= 5.0 and 1 <= soh <= 8:
                rul_prediction_int = int(rul_prediction_int * (kapasitas_nominal / 4973))
        elif battery_type == '18650':
            if 4.9 <= voltage <= 5.0 and 1 <= soh <= 8:
                rul_prediction_int = int(rul_prediction_int * (kapasitas_nominal / 2955))

        return str(rul_prediction_int)[:2], rul_prediction_int, kapasitas_mAh
    
    except Exception as e:
        logging.error(f"Error in predict_rul: {e}") 
        raise

def predict_soh(voltage):
    try:
        X_manual_reshaped = preprocess_input_data_soh(voltage, choicemodelsoh)
        if choicemodelsoh == 'lstm':
            soh_prediction_scaled = model_soh_lstm.predict(X_manual_reshaped)
        elif choicemodelsoh == 'dtree':
            soh_prediction_scaled = model_soh_dtree.predict(X_manual_reshaped)
        elif choicemodelsoh == 'svr':
            soh_prediction_scaled = model_soh_svr.predict(X_manual_reshaped)
        else:
            raise ValueError(f"Invalid model choice for SOH prediction: {choicemodelsoh}")

        soh_prediction = scaler_y_soh.inverse_transform(soh_prediction_scaled.reshape(-1, 1)).flatten()[0]
        soh_prediction = min(max(int(soh_prediction), 0), 100)  # Konversi ke persentase dan batasi rentang

        # Menentukan kapasitas baterai
        kapasitas_nominal = 4973 if battery_type == '21700' else 2955
        kapasitas_mAh = (soh_prediction / 100) * kapasitas_nominal

        return str(soh_prediction), soh_prediction, kapasitas_mAh
    except Exception as e:
        logging.error(f"Error in predict_soh: {e}")  
        raise

@app.route("/", methods=['GET'])
def index():
    return render_template("starting.html")

@app.route("/baterai", methods=['GET', 'POST'])
def baterai():
    global battery_type
    if request.method == 'POST':
        battery_type = request.form.get('battery_type')
        if battery_type not in ['18650', '21700']:
            return jsonify({'error': 'Invalid battery type selected'}), 400
        load_models(battery_type)
        return redirect(url_for('menu'))
    return render_template("baterai.html")

@app.route("/menu", methods=['GET'])
def menu():
    return render_template("index.html")

@app.route('/graph', methods=['GET'])
def graph():
    data_file = 'datahasil/datahasil.csv' if battery_type == '18650' else 'datahasil/datahasil2.csv'
    try:
        data = pd.read_csv(data_file)
        data = data[['Voltage', 'Temperature', 'SOH', 'RUL']]
        data_json = data.to_json(orient='records')
    except FileNotFoundError as e:
        logging.error(f"Error loading data file: {e}")
        return jsonify({'error': 'Data file not found'}), 404
    return render_template('graph.html', data=data_json)

@app.route("/modeloption", methods=['GET', 'POST'])
def modeloption():
    global choicemodel
    if request.method == 'POST':
        choicemodel = request.form.get('model')
        if choicemodel not in ['lstm', 'dtree', 'svr']:
            return jsonify({'error': 'Invalid model choice'}), 400
        return redirect(url_for('predrul'))
    return render_template("modeloption.html")

@app.route("/modeloptionsoh", methods=['GET', 'POST'])
def modeloptionsoh():
    global choicemodelsoh
    if request.method == 'POST':
        choicemodelsoh = request.form.get('modelsoh')
        if choicemodelsoh not in ['lstm', 'dtree', 'svr']:
            return jsonify({'error': 'Invalid model choice for SOH'}), 400
        return redirect(url_for('predsoh'))
    return render_template("modeloptionsoh.html")

@app.route("/predrul", methods=['GET', 'POST'])
def predrul():
    if request.method == 'POST':
        try:
            soh = float(request.form.get('soh'))
            voltage = float(request.form.get('voltage'))

            if not (3.4 <= voltage <= 4.2):
                return render_template('warning.html', message='Input voltage salah. Harus antara 3.4 dan 4.2.')

            predicted_rul, rul, kapasitas_mAh = predict_rul(soh, voltage)
            temperatur = 24.2 if rul > 7000 else 24 if rul > 4000 else 23.8
            
            return render_template('result.html', soh=soh, voltage=voltage, temperatur=temperatur, predicted_rul=predicted_rul, kapasitas=battery_type, kapasitas_mAh=kapasitas_mAh)
        except Exception as e:
            logging.error(f"Error in predrul: {e}")
            return jsonify({'error': str(e)}), 500
    return render_template("cekrul.html", model=choicemodel.capitalize() if choicemodel else 'Model')

@app.route("/predsoh", methods=['GET', 'POST'])
def predsoh():
    if request.method == 'POST':
        try:
            voltage = float(request.form.get('voltage'))

            if not (3.4 <= voltage <= 4.2):
                return render_template('warning.html', message='Input voltage salah. Harus antara 3.4 dan 4.2.')

            predicted_soh, soh, kapasitas_mAh = predict_soh(voltage)
            temperatur = predict_soh_temperature(predicted_soh)
            
            return render_template('resultsoh.html', voltage=voltage, temperatur=temperatur, predicted_soh=predicted_soh, kapasitas_mAh=kapasitas_mAh)
        except Exception as e:
            logging.error(f"Error in predsoh: {e}")  
            return jsonify({'error': str(e)}), 500
    return render_template("ceksoh.html")

if __name__ == '__main__':
    app.run(debug=True)
