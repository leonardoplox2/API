import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# Ruta segura al archivo CSV (usa raw string)
ruta_csv = r"C:\Users\PC-01\API\ALGORITMO (SVM)\DATASET_RIOSAVILA_EC1.csv"

# Verificar si el archivo existe
if not os.path.exists(ruta_csv):
    print(f"❌ No se encontró el archivo: {ruta_csv}")
    exit()

# Cargar dataset
df = pd.read_csv(ruta_csv)

# Eliminar columna ID si existe
if "ID" in df.columns:
    df = df.drop("ID", axis=1)

# Separar características y etiqueta
X = df.drop("RESULT", axis=1)
y = df["RESULT"]

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo SVM
model = SVC()
model.fit(X_scaled, y)

# Guardar modelos entrenados
with open("modelo_entrenado.pkl", "wb") as f:
    pickle.dump(model, f)

with open("escalador.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Modelo y escalador entrenados correctamente y guardados.")
