from flask import Flask, request, jsonify, send_from_directory, redirect
import pickle
import pandas as pd
import os
import webbrowser
import threading

app = Flask(__name__)

# Cargar el modelo entrenado
with open('modelo_entrenado.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

# Página de bienvenida
@app.route('/')
def index():
    return redirect('/formulario')

# Servir el formulario HTML
@app.route('/formulario')
def formulario():
    return send_from_directory(os.path.join(os.getcwd(), 'public'), 'index.html')

# Ruta para archivos estáticos como CSS o JS
@app.route('/public/<path:filename>')
def public_files(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'public'), filename)

# Ruta para predecir riesgo
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])

        # Validación de columnas
        expected_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        if not all(col in input_data.columns for col in expected_columns):
            return jsonify({'error': '❌ Las columnas de entrada no coinciden con las esperadas.'}), 400

        # Predicción
        prediction = modelo.predict(input_data)[0]

        # Interpretación
        if prediction == 1:
            resultado = 'Bajo riesgo'
        elif prediction == 2:
            resultado = 'Riesgo intermedio'
        elif prediction == 3:
            resultado = 'Alto riesgo'
        else:
            resultado = 'Desconocido'

        return jsonify({
            'predicción': resultado,
            'valor_numérico': int(prediction)
        })

    except Exception as e:
        return jsonify({'error': f'❌ Error en la predicción: {str(e)}'}), 500

# Función para iniciar Flask
def iniciar_app():
    app.run(port=5000, debug=False)

# Iniciar el servidor Flask en un hilo y abrir el navegador
if __name__ == '__main__':
    # Crear un hilo para ejecutar el servidor Flask
    thread = threading.Thread(target=iniciar_app)
    thread.start()

    # Abrir el navegador automáticamente en la URL
    webbrowser.open("http://127.0.0.1:5000/")



