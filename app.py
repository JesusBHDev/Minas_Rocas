from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        freq_11 = float(request.form['freq_11'])
        freq_24 = float(request.form['freq_24'])
        freq_31 = float(request.form['freq_31'])
        freq_32 = float(request.form['freq_32'])
        freq_36 = float(request.form['freq_36'])
        freq_40 = float(request.form['freq_40'])
        freq_43 = float(request.form['freq_43'])
        freq_45 = float(request.form['freq_45'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[freq_11, freq_24, freq_31, freq_32, freq_36, freq_40, freq_43, freq_45]], 
                               columns=['Freq_11', 'Freq_24', 'Freq_31', 'Freq_32', 'Freq_36', 'Freq_40', 'Freq_43', 'Freq_45'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
