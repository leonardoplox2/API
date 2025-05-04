from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
import os

app = Flask(__name__, static_folder="public")

# Cargar modelo y escalador una sola vez al iniciar el servidor
with open("modelo/modelo_entrenado.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("modelo/escalador.pkl", "rb") as f:
    escalador = pickle.load(f)

# Ruta para servir la página principal
@app.route("/")
def index():
    return send_from_directory("public", "index.html")

# Ruta para servir archivos estáticos (JS, CSS, etc.)
@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("public", path)

# Ruta para predicción
@app.route("/prediccion", methods=["POST"])
def prediccion():
    try:
        datos = request.json

        # Verificar que estén todas las columnas necesarias
        columnas_esperadas = [
            "THINCKNESS", "SIZE", "SHAPE", "ADHESION",
            "SINGLE", "NUCLEI", "CHROMATIN", "NUCLEOLI", "MITOSIS"
        ]

        # Crear DataFrame asegurando el orden correcto
        df = pd.DataFrame([[datos.get(col) for col in columnas_esperadas]], columns=columnas_esperadas)

        # Escalar y predecir
        df_scaled = escalador.transform(df)
        prediccion = modelo.predict(df_scaled)

        resultado = {
            "resultado": "Maligno ⚠️" if prediccion[0] == 1 else "Benigno ✅",
            "valor": int(prediccion[0])
        }

        return jsonify(resultado)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
