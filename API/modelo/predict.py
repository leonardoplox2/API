import sys
import io
import json
import pickle
import pandas as pd
import os

try:
    entrada = json.load(io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8-sig'))

    df = pd.DataFrame([entrada])

    with open("modelo_entrenado.pkl", "rb") as f:
        modelo = pickle.load(f)

    with open("escalador.pkl", "rb") as f:
        escalador = pickle.load(f)

    df_scaled = escalador.transform(df)
    prediccion = modelo.predict(df_scaled)

    resultado = {
        "resultado": "Maligno ⚠️" if prediccion[0] == 1 else "Benigno ✅",
        "valor": int(prediccion[0])
    }

    # SOLO JSON limpio
    print(json.dumps(resultado))

except Exception as e:
    print(json.dumps({"error": str(e)}))
