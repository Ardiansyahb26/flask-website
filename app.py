import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

app = Flask(__name__)
choicemodel = ''
choicemodelsoh = ''
battery_type = '18650'

def load_models(battery_type):
    global model_rul_lstm, scaler_rul_lstm, model_rul_dtree, scaler_rul_dtree, model_rul_svr, scaler_rul_svr
    global model_soh_lstm, scaler_soh_lstm, model_soh_dtree, scaler_soh_dtree, model_soh_svr, scaler_soh_svr
    global scaler_y_rul, scaler_y_soh

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
    scaler_y_rul.fit(np.array([0, 100]).reshape(-1, 1))

    scaler_y_soh = MinMaxScaler()
    scaler_y_soh.fit(np.array([0, 1]).reshape(-1, 1))

load_models(battery_type)

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
        return X_manual_reshaped
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
        rul_prediction_str = str(rul_prediction_int)[:2]
        

        if soh > 99:
            temperature = 25.0  # Baterai sangat baru, performa optimal
        elif 98 < soh <= 99:
            temperature = 24.9  # Baterai baru, performa sangat baik
        elif 97 < soh <= 98:
            temperature = 24.8  # Kondisi sangat baik, degradasi sangat sedikit
        elif 96 < soh <= 97:
            temperature = 24.7  # Performanya sangat baik, degradasi minimal
        elif 95 < soh <= 96:
            temperature = 24.6  # Baterai dalam kondisi baik, degradasi mulai terlihat
        elif 94 < soh <= 95:
            temperature = 24.5  # Kondisi hampir optimal, degradasi kecil
        elif 93 < soh <= 94:
            temperature = 24.4  # Performanya baik, sedikit penurunan
        elif 92 < soh <= 93:
            temperature = 24.3  # Degradasi ringan, kondisi masih sangat baik
        elif 91 < soh <= 92:
            temperature = 24.2  # Degradasi ringan, performa masih baik
        elif 90 < soh <= 91:
            temperature = 24.1  # Baterai dalam kondisi baik, degradasi kecil
        elif 88 < soh <= 90:
            temperature = 24.0  # Performanya baik, degradasi mulai terasa
        elif 86 < soh <= 88:
            temperature = 23.9  # Kondisi cukup baik, degradasi mulai signifikan
        elif 84 < soh <= 86:
            temperature = 23.8  # Performanya menurun sedikit, degradasi jelas
        elif 82 < soh <= 84:
            temperature = 23.7  # Kondisi baterai menurun, degradasi mulai mempengaruhi
        elif 80 < soh <= 82:
            temperature = 23.6  # Performanya menurun, degradasi signifikan
        elif 78 < soh <= 80:
            temperature = 23.5  # Baterai menunjukkan degradasi, suhu mulai menurun
        elif 76 < soh <= 78:
            temperature = 23.4  # Performanya terpengaruh, degradasi cukup besar
        elif 74 < soh <= 76:
            temperature = 23.3  # Kondisi menurun, degradasi jelas terasa
        elif 72 < soh <= 74:
            temperature = 23.2  # Performanya semakin menurun, degradasi lebih besar
        elif 70 < soh <= 72:
            temperature = 23.1  # Kondisi baterai mulai menurun drastis, degradasi signifikan
        elif 68 < soh <= 70:
            temperature = 23.0  # Performanya menurun, degradasi semakin parah
        elif 66 < soh <= 68:
            temperature = 22.9  # Degradasi terlihat jelas, performa terpengaruh
        elif 64 < soh <= 66:
            temperature = 22.8  # Kondisi baterai menurun drastis, degradasi besar
        elif 62 < soh <= 64:
            temperature = 22.7  # Performanya terpengaruh, degradasi signifikan
        elif 60 < soh <= 62:
            temperature = 22.6  # Kondisi baterai semakin menurun, degradasi besar
        elif 58 < soh <= 60:
            temperature = 22.5  # Performanya menurun drastis, degradasi sangat jelas
        elif 56 < soh <= 58:
            temperature = 22.4  # Kondisi baterai buruk, degradasi besar
        elif 54 < soh <= 56:
            temperature = 22.3  # Performanya terpengaruh, degradasi besar
        elif 52 < soh <= 54:
            temperature = 22.2  # Kondisi baterai menurun drastis, degradasi parah
        elif 50 < soh <= 52:
            temperature = 22.1  # Performanya buruk, degradasi signifikan
        elif 48 < soh <= 50:
            temperature = 22.0  # Kondisi baterai buruk, degradasi sangat besar
        elif 46 < soh <= 48:
            temperature = 21.9  # Performanya menurun sangat besar, degradasi besar
        elif 44 < soh <= 46:
            temperature = 21.8  # Kondisi baterai sangat buruk, degradasi besar
        elif 42 < soh <= 44:
            temperature = 21.7  # Performanya buruk, degradasi sangat besar
        elif 40 < soh <= 42:
            temperature = 21.6  # Kondisi baterai hampir mati, degradasi sangat parah
        elif 38 < soh <= 40:
            temperature = 21.5  # Performanya buruk, degradasi besar
        elif 36 < soh <= 38:
            temperature = 21.4  # Kondisi baterai sangat buruk, degradasi besar
        elif 34 < soh <= 36:
            temperature = 21.3  # Performanya hampir tidak ada, degradasi sangat besar
        elif 32 < soh <= 34:
            temperature = 21.2  # Kondisi baterai sangat buruk, hampir tidak bisa digunakan
        elif 30 < soh <= 32:
            temperature = 21.1  # Performanya hampir habis, degradasi sangat besar
        elif 28 < soh <= 30:
            temperature = 21.0  # Kondisi baterai hampir tidak bisa digunakan, degradasi sangat besar
        elif 26 < soh <= 28:
            temperature = 20.9  # Performanya hampir nol, degradasi sangat besar
        elif 24 < soh <= 26:
            temperature = 20.8  # Kondisi baterai sangat buruk, degradasi sangat parah
        elif 22 < soh <= 24:
            temperature = 20.7  # Performanya hampir habis, degradasi besar
        elif 20 < soh <= 22:
            temperature = 20.6  # Kondisi baterai hampir mati, degradasi sangat besar
        elif 18 < soh <= 20:
            temperature = 20.5  # Performanya hampir tidak ada, degradasi besar
        elif 16 < soh <= 18:
            temperature = 20.4  # Kondisi baterai sangat buruk, degradasi besar
        elif 14 < soh <= 16:
            temperature = 20.3  # Performanya hampir nol, degradasi besar
        elif 12 < soh <= 14:
            temperature = 20.2  # Kondisi baterai hampir tidak bisa digunakan, degradasi besar
        elif 10 < soh <= 12:
            temperature = 20.1  # Performanya sangat buruk, degradasi besar
        elif 8 < soh <= 10:
            temperature = 20.0  # Kondisi baterai hampir mati, degradasi sangat besar
        elif 6 < soh <= 8:
            temperature = 19.9  # Performanya hampir tidak ada, degradasi sangat besar
        elif 4 < soh <= 6:
            temperature = 19.8  # Kondisi baterai hampir habis, degradasi besar
        elif 2 < soh <= 4:
            temperature = 19.7  # Performanya sangat buruk, degradasi besar
        else:
            temperature = 19.6  # Baterai mati, tidak ada performa yang tersisa
            

        return rul_prediction_str, temperature
    
    except Exception as e:
        print(f"Error in predict_rul: {e}") 
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
             
        soh_prediction_str = str(soh_prediction)  
        return soh_prediction_str
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

            if voltage > 4.2 or voltage < 3.4:
                return render_template('warning.html', message='Input voltage salah. Harus antara 3.4 dan 4.2.')

            predicted_rul, temperature = predict_rul(soh, voltage)
            
            return render_template('result.html', soh=soh, voltage=voltage, predicted_rul=predicted_rul, temperature=temperature)
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

            if voltage > 4.2 or voltage < 3.4:
                return render_template('warning.html', message='Input voltage salah. Harus antara 3.4 dan 4.2.')

            predicted_soh = predict_soh(voltage)

            return render_template('resultsoh.html', voltage=voltage, predicted_soh=predicted_soh)
        except Exception as e:
            return jsonify({'error': str(e)})
    elif request.method == 'GET':
        return render_template('ceksoh.html')
    return render_template("ceksoh.html")

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1')
