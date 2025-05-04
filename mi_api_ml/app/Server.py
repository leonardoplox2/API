from flask import Flask, request, jsonify, send_from_directory
from prediccion import predecir
import os

app = Flask(__name__, static_folder="../public")

@app.route("/")
def index():
    return send_from_directory("../public", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("../public", path)

@app.route("/prediccion", methods=["POST"])
def ruta_prediccion():
    try:
        datos = request.json
        texto, valor = predecir(datos)
        return jsonify({"resultado": texto, "valor": valor})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)

