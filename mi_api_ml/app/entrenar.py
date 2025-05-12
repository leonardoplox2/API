import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# 1. Cargar el dataset
dataframe = pd.read_csv('Data_limpia_Maternal_Risk_base_de_datos.csv', sep=',')

# 2. Limpiar datos: quitar infinitos y NaNs
dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
dataframe.dropna(inplace=True)

# 3. No es necesario mapear, ya que 'RiskLevel' ya contiene valores numéricos
# Si quisieras, aquí puedes verificar la columna 'RiskLevel'
print("Valores únicos en 'RiskLevel':", dataframe['RiskLevel'].unique())

# 4. Separar variables independientes y dependiente
X = dataframe.drop('RiskLevel', axis=1)
y = dataframe['RiskLevel']

# 5. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

# 6. Entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=80, random_state=42)
modelo.fit(X_train, y_train)

# 7. Evaluación del modelo
y_pred = modelo.predict(X_test)
print("\n📊 Reporte de clasificación:\n")
print(classification_report(y_test, y_pred, target_names=['Bajo', 'Intermedio', 'Alto']))

# 8. Guardar el modelo entrenado
with open('modelo_entrenado.pkl', 'wb') as archivo_salida:
    pickle.dump(modelo, archivo_salida, protocol=pickle.HIGHEST_PROTOCOL)

print("\n✅ Modelo entrenado y guardado exitosamente como 'modelo_entrenado.pkl'")

