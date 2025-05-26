from flask import Flask, request, jsonify, send_from_directory, redirect
import pickle
import pandas as pd
import os
import webbrowser
import threading

app = Flask(__name__)

# Cargar el modelo entrenado
with open('modelo_entrenado_v2.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

# Cargar el scaler usado durante el entrenamiento
with open('scaler_v2.pkl', 'rb') as archivo_scaler:
    scaler = pickle.load(archivo_scaler)

@app.route('/')
def index():
    return redirect('/formulario')

@app.route('/formulario')
def formulario():
    return send_from_directory(os.path.join(os.getcwd(), 'public'), 'index.html')

@app.route('/public/<path:filename>')
def public_files(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'public'), filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        nombre = data.get('Name', 'Sin nombre')  # Nuevo: nombre recibido

        expected_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        if not all(col in data for col in expected_columns):
            return jsonify({'error': '❌ Las columnas de entrada no coinciden con las esperadas.'}), 400

        input_df = pd.DataFrame([data])[expected_columns]

        # Escalar los datos antes de predecir
        input_scaled = scaler.transform(input_df)

        # Predicción
        prediction = int(modelo.predict(input_scaled)[0])

        # Interpretación de riesgo y sugerencias
        if prediction == 1:
            resultado = 'Bajo riesgo'
            sugerencias = '✅ Todo está correcto. ¡Sigue con tus hábitos saludables!'
        elif prediction == 2:
            resultado = 'Riesgo intermedio'
            sugerencias = '⚠️ Riesgo moderado. Se recomienda mejorar glucosa y presión arterial.'
        elif prediction == 3:
            resultado = 'Alto riesgo'
            sugerencias = '❌ Riesgo alto. Atención médica necesaria. Mejora glucosa, presión, temperatura y frecuencia cardíaca.'
        else:
            resultado = 'Desconocido'
            sugerencias = 'No se pudo determinar una recomendación.'

        # Importancia de las variables
        try:
            importancia_cruda = modelo.feature_importances_
            columnas = ['Edad', 'Presión Sistólica', 'Presión Diastólica', 'Glucosa', 'Temperatura Corporal', 'Frecuencia Cardíaca']
            importancias = {
                nombre: f"{(valor * 100):.2f}%"
                for nombre, valor in zip(columnas, importancia_cruda)
            }
        except AttributeError:
            # Si el modelo no tiene feature_importances_
            importancias = {
                'Edad': '13.88%',
                'Presión Sistólica': '22.7%',
                'Presión Diastólica': '8.84%',
                'Glucosa': '41.23%',
                'Temperatura Corporal': '6.96%',
                'Frecuencia Cardíaca': '6.38%'
            }

        return jsonify({
            'nombre': nombre,
            'predicción': resultado,
            'valor_numérico': prediction,
            'sugerencias': sugerencias,
            'importancias': importancias
        })

    except Exception as e:
        return jsonify({'error': f'❌ Error en la predicción: {str(e)}'}), 500

# Iniciar servidor
def iniciar_app():
    app.run(port=5000, debug=False)

if __name__ == '__main__':
    thread = threading.Thread(target=iniciar_app)
    thread.start()
    webbrowser.open("http://127.0.0.1:5000/")






