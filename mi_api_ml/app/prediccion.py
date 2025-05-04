import pickle
import pandas as pd

# Cargar modelo y escalador
with open("../modelo/modelo_entrenado.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("../modelo/escalador.pkl", "rb") as f:
    escalador = pickle.load(f)

# Función de predicción
def predecir(data_dict):
    columnas = [
        "THINCKNESS", "SIZE", "SHAPE", "ADHESION",
        "SINGLE", "NUCLEI", "CHROMATIN", "NUCLEOLI", "MITOSIS"
    ]
    df = pd.DataFrame([[data_dict[col] for col in columnas]], columns=columnas)
    df_scaled = escalador.transform(df)
    pred = modelo.predict(df_scaled)
    return "Maligno ⚠️" if pred[0] == 1 else "Benigno ✅", int(pred[0])

