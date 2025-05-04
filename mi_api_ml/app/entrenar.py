# entrenar_modelo.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import os

# Cargar dataset
df = pd.read_csv(r"C:\Users\PC-01\API\ALGORITMO (SVM)\DATASET_RIOSAVILA_EC1.csv")


if "ID" in df.columns:
    df = df.drop("ID", axis=1)


X = df.drop("RESULT", axis=1)
y = df["RESULT"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo SVM
model = SVC()
model.fit(X_train_scaled, y_train)

# Evaluar precisión
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Precisión del modelo SVM: {accuracy:.2%}")

# Crear carpeta 'modelo' si no existe
os.makedirs("modelo", exist_ok=True)

# Guardar modelo y escalador
with open("modelo/modelo_entrenado.pkl", "wb") as f:
    pickle.dump(model, f)

with open("modelo/escalador.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Modelo SVM y escalador guardados correctamente.")
