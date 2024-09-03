import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)
choicemodel = ''
choicemodelsoh = ''
battery_type = '18650'

soh=0
rul=0


def load_models(battery_type):
    global model_rul_lstm, scaler_rul_lstm, model_rul_dtree, scaler_rul_dtree, model_rul_svr, scaler_rul_svr
    global model_soh_lstm, scaler_soh_lstm, model_soh_dtree, scaler_soh_dtree, model_soh_svr, scaler_soh_svr
    global scaler_y_rul, scaler_y_soh, scaler_y_rul_temperatur, scaler_y_soh_temperatur
    global temperature_model_rul_lstm, temperatur_scaler_rul_lstm, temperature_model_rul_dtree, temperatur_scaler_rul_dtree, temperature_model_rul_svr, temperatur_scaler_rul_svr
    global temperature_model_soh_lstm, temperatur_scaler_soh_lstm, temperature_model_soh_dtree, temperatur_scaler_soh_dtree, temperature_model_soh_svr, temperatur_scaler_soh_svr

    model_rul_lstm = joblib.load(f'model/{battery_type}/lstm/lstm_model.pkl')
    scaler_rul_lstm = joblib.load(f'model/{battery_type}/lstm/lstm_scaler.pkl')

    model_rul_dtree = joblib.load(f'model/{battery_type}/dtree/dtree.pkl')
    scaler_rul_dtree = joblib.load(f'model/{battery_type}/dtree/scalerdtree.pkl')

    model_rul_svr = joblib.load(f'model/{battery_type}/svr/svr.pkl')
    scaler_rul_svr = joblib.load(f'model/{battery_type}/svr/scalersvr.pkl')

    model_soh_lstm = joblib.load(f'modelsoh/{battery_type}/lstm.pkl')
    scaler_soh_lstm = joblib.load(f'modelsoh/{battery_type}/lstmscaler.pkl')

    model_soh_dtree = joblib.load(f'modelsoh/{battery_type}/dtree.pkl')
    scaler_soh_dtree = joblib.load(f'modelsoh/{battery_type}/scalerdtree.pkl')

    model_soh_svr = joblib.load(f'modelsoh/{battery_type}/svr.pkl')
    scaler_soh_svr = joblib.load(f'modelsoh/{battery_type}/scalersvr.pkl')

    scaler_y_rul = MinMaxScaler()
    scaler_y_rul.fit(np.array([0, 100]).reshape(-1, 1))  # Assuming RUL is between 0 and 100 for scaling back

    scaler_y_soh = MinMaxScaler()
    scaler_y_soh.fit(np.array([0, 1]).reshape(-1, 1))  # Assuming SOH is between 0 and 1 for scaling back

 #=============================================
    
    temperature_model_rul_lstm = joblib.load(f'model/modeltemperatur/lstm/lstm_model.pkl')
    temperature_scaler_rul_lstm = joblib.load(f'model/modeltemperatur/lstm/lstm_scaler.pkl')

    temperature_model_rul_dtree = joblib.load(f'model/modeltemperatur/dtree/dtree.pkl')
    temperature_scaler_rul_dtree = joblib.load(f'model/modeltemperatur/dtree/scalerdtree.pkl')

    temperature_model_rul_svr = joblib.load(f'model/modeltemperatur/svr/svr.pkl')
    temperature_scaler_rul_svr = joblib.load(f'model/modeltemperatur/svr/scalersvr.pkl')

    temperature_model_soh_lstm = joblib.load(f'modelsoh/modeltemperatur/lstm/lstm_model.pkl')
    temperature_scaler_soh_lstm = joblib.load(f'modelsoh/modeltemperatur/lstm/lstm_scaler.pkl')

    temperature_model_soh_dtree = joblib.load(f'modelsoh/modeltemperatur/dtree/dtree.pkl')
    temperature_scaler_soh_dtree = joblib.load(f'modelsoh/modeltemperatur/dtree/scalerdtree.pkl')

    temperature_model_soh_svr = joblib.load(f'modelsoh/modeltemperatur/svr/svr.pkl')
    temperature_scaler_soh_svr = joblib.load(f'modelsoh/modeltemperatur/svr/scalersvr.pkl')

    scaler_y_temperatur = MinMaxScaler()
    scaler_y_temperatur.fit(np.array([0, 100]).reshape(-1, 1))  # Assuming RUL is between 0 and 100 for scaling back

    scaler_y_soh_temperatur = MinMaxScaler()
    scaler_y_soh_temperatur.fit(np.array([0, 1]).reshape(-1, 1))  # Assuming SOH is between 0 and 1 for scaling back


load_models(battery_type)

def preprocess_input_data_soh_temperatur(soh, model_type):
    X_manual = np.array([[soh]])
    if model_type == 'lstm':
        X_manual_scaled = scaler_soh_lstm.transform(X_manual)
        X_manual_reshaped = X_manual_scaled.reshape((X_manual_scaled.shape[0], 1, X_manual_scaled.shape[1]))
        return X_manual_scaled
    elif model_type == 'dtree':
        X_manual_scaled = scaler_soh_dtree.transform(X_manual)
        return X_manual_scaled
    elif model_type == 'svr':
        X_manual_scaled = scaler_soh_svr.transform(X_manual)
        return X_manual_scaled

def predict_soh_temperature(soh):
    try:
        print(soh)
        soh_prediction_scaled = "";
        X_manual_reshaped = preprocess_input_data_soh_temperatur(soh,choicemodelsoh)
        if choicemodelsoh == 'lstm':
            soh_prediction_scaled = temperature_model_soh_lstm.predict(X_manual_reshaped)
        elif choicemodelsoh == 'dtree':
            soh_prediction_scaled = temperature_model_soh_dtree.predict(X_manual_reshaped)
        elif choicemodelsoh == 'svr':
            soh_prediction_scaled = temperature_model_soh_svr.predict(X_manual_reshaped)
        else:
            raise ValueError(f"Invalid model choice for RUL prediction: {choicemodel}")

        soh_prediction = scaler_y_soh_temperatur.inverse_transform(soh_prediction_scaled.reshape(-1, 1)).flatten()[0]
        soh_prediction_int = int(soh_prediction)  
        # if soh_prediction < 0 :
        #     soh_prediction = 0
        # elif soh_prediction > 100 :
        #     soh_prediction = 100
        # if voltage >= 4.1 and soh_prediction >= 1 and choicemodelsoh == 'lstm':
        #     soh_prediction = 100
        #     print(soh_prediction)
            
        soh_prediction_str = str(soh_prediction)[:4]
        return soh_prediction_str
    
    except Exception as e:
        print(f"Error in predict_rul: {e}") 
        raise

# utk preprocess data
def preprocess_input_data_rul(soh, voltage, model_type):
    X_manual = np.array([[soh, voltage]])
    
    if model_type == 'lstm':
        X_manual_scaled = scaler_rul_lstm.transform(X_manual)
        X_manual_reshaped = X_manual_scaled.reshape((X_manual_scaled.shape[0], 1, X_manual_scaled.shape[1]))
        return X_manual_reshaped
    elif model_type == 'dtree':
        X_manual_scaled = scaler_rul_dtree.transform(X_manual)
        return X_manual_scaled
    elif model_type == 'svr':
        X_manual_scaled = scaler_rul_svr.transform(X_manual)
        return X_manual_scaled
    
def preprocess_input_data_soh(voltage, model_type):
    X_manual = np.array([[voltage]])
    
    if model_type == 'lstm':
        X_manual_scaled = scaler_soh_lstm.transform(X_manual)
        X_manual_reshaped = X_manual_scaled.reshape((X_manual_scaled.shape[0], 1, X_manual_scaled.shape[1]))
        return X_manual_scaled
    elif model_type == 'dtree':
        X_manual_scaled = scaler_soh_dtree.transform(X_manual)
        return X_manual_scaled
    elif model_type == 'svr':
        X_manual_scaled = scaler_soh_svr.transform(X_manual)
        return X_manual_scaled


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
        rul_prediction_int = int(rul_prediction)  
        if rul_prediction_int < 0 :
            rul_prediction_int = 0
        rul = rul_prediction_int
        
        print("rul awal : ")
        print(rul)
        rul_prediction_str = str(rul_prediction_int)[:2]
        return rul_prediction_str,rul
    
    except Exception as e:
        print(f"Error in predict_rul: {e}") 
        raise




def predict_soh(voltage):
    try:
        X_manual_reshaped = preprocess_input_data_soh(voltage, choicemodelsoh)
        soh_prediction_scaled = ""
        if choicemodelsoh == 'lstm':
            soh_prediction_scaled = model_soh_lstm.predict(X_manual_reshaped)
        elif choicemodelsoh == 'dtree':
            soh_prediction_scaled = model_soh_dtree.predict(X_manual_reshaped)
        elif choicemodelsoh == 'svr':
            soh_prediction_scaled = model_soh_svr.predict(X_manual_reshaped)
        else:
            raise ValueError(f"Invalid model choice for SOH prediction: {choicemodelsoh}")

        soh_prediction = scaler_y_soh.inverse_transform(soh_prediction_scaled.reshape(-1, 1)).flatten()[0]

        if soh_prediction < 1:
            soh_prediction = int(soh_prediction * 100)
        else:
            soh_prediction = int(soh_prediction) 

        if soh_prediction < 0 :
            soh_prediction = 0
        elif soh_prediction > 100 :
            soh_prediction = 100
        if voltage >= 4.1 and soh_prediction >= 1 and choicemodelsoh == 'lstm':
            soh_prediction = 100
            print(soh_prediction)
                
        soh = soh_prediction
        print("soh : ")
        print(soh)
        soh_prediction_str = str(soh_prediction)  
        return soh_prediction_str,soh
    except Exception as e:
        print(f"Error in predict_soh: {e}")  
        raise




@app.route("/", methods=['GET'])
def index():
    return render_template("starting.html")

@app.route("/baterai", methods=['GET', 'POST'])
def baterai():
    global battery_type
    if request.method == 'POST':
        battery_type = request.form['battery_type']
        load_models(battery_type)
        return redirect(url_for('menu'))
    return render_template("baterai.html")

@app.route("/menu", methods=['GET'])
def menu():
    return render_template("index.html")

@app.route('/graph', methods=['GET'])
def graph():
    data_file = 'datahasil/datahasil.csv' if battery_type == '18650' else 'datahasil/datahasil2.csv'
    data = pd.read_csv(data_file)
    columns = ['Voltage', 'Temperature', 'SOH', 'RUL']
    data = data[columns]
    data_json = data.to_json(orient='records')
    
    return render_template('graph.html', data=data_json)

@app.route("/modeloption", methods=['GET', 'POST'])
def modeloption():
    global choicemodel
    if request.method == 'POST':
        try:
            choicemodel = request.form['model']
            return redirect(url_for('predrul'))
        except Exception as e:
            return jsonify({'error': str(e)})
    return render_template("modeloption.html")

@app.route("/modeloptionsoh", methods=['GET', 'POST'])
def modeloptionsoh():
    global choicemodelsoh
    if request.method == 'POST':
        try:
            choicemodelsoh = request.form['modelsoh']
            return redirect(url_for('predsoh'))
        except Exception as e:
            return jsonify({'error': str(e)})
    return render_template("modeloptionsoh.html")

@app.route("/predrul", methods=['GET', 'POST'])
def predrul():
    global choicemodel
    if request.method == 'POST':
        try:
            soh = float(request.form['soh'])
            voltage = float(request.form['voltage'])
            temperatur=0
            # waktu_pengosongan = float(request.form['waktu_pengosongan'])

            # Validasi voltage
            if voltage > 4.2 or voltage < 3.4:
                return render_template('warning.html', message='Input voltage salah. Harus antara 3.4 dan 4.2.')

            predicted_rul,rul = predict_rul(soh, voltage)
            
            print("rul : ")
            print(rul)
            if rul > 7000:
                temperatur = 24.2
            elif rul <=7000 and rul > 4000:
                temperatur = 24
            elif rul <= 4000:
                temperatur = 23.8
            
            # if predicted_rul < 0:
            #     predict_rul = 0;
            return render_template('result.html', soh=soh, voltage=voltage,temperatur=temperatur, predicted_rul=predicted_rul)
        except Exception as e:
            return jsonify({'error': str(e)})
    elif request.method == 'GET':
        if choicemodel == 'lstm':
            return render_template('cekrul.html', model='LSTM')
        elif choicemodel == 'dtree':
            return render_template('cekrul.html', model='Decision Tree')
        elif choicemodel == 'svr':
            return render_template('cekrul.html', model='SVR')
    return render_template("cekrul.html")


@app.route("/predsoh", methods=['GET', 'POST'])
def predsoh():
    global choicemodelsoh
    if request.method == 'POST':
        try:
            voltage = float(request.form['voltage'])
            temperatur = 0

            # Validasi voltage
            if voltage > 4.2 or voltage < 3.4:
                return render_template('warning.html', message='Input voltage salah. Harus antara 3.4 dan 4.2.')

            predicted_soh,soh = predict_soh(voltage)
            print("prediksi soh : ")
            print(predicted_soh)
            temperatur = predict_soh_temperature(predicted_soh)
            print("temperature : ")
            print(temperatur)
            
            # print("soh")
            # print(soh)
            # if soh > 70:
            #     temperatur = 24.2
            # elif soh <=70 and soh > 40:
            #     temperatur = 24
            # elif soh <= 40:
            #     temperatur = 23.8

            return render_template('resultsoh.html', voltage=voltage,temperatur=temperatur, predicted_soh=predicted_soh)
        except Exception as e:
            return jsonify({'error': str(e)})
    elif request.method == 'GET':
        return render_template('ceksoh.html')
    return render_template("ceksoh.html")


if __name__ == '__main__':
    app.run(debug=True)
