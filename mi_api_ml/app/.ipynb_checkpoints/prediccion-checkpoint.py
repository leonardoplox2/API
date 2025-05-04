{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2db122f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "import pandas as pd\n",
    "\n",
    "with open(\"../modelo/modelo_entrenado.pkl\", \"rb\") as f:\n",
    "    modelo = pickle.load(f)\n",
    "\n",
    "with open(\"../modelo/escalador.pkl\", \"rb\") as f:\n",
    "    escalador = pickle.load(f)\n",
    "\n",
    "def predecir(data_dict):\n",
    "    columnas = [\"THINCKNESS\", \"SIZE\", \"SHAPE\", \"ADHESION\", \"SINGLE\", \"NUCLEI\", \"CHROMATIN\", \"NUCLEOLI\", \"MITOSIS\"]\n",
    "    df = pd.DataFrame([[data_dict[col] for col in columnas]], columns=columnas)\n",
    "    df_scaled = escalador.transform(df)\n",
    "    pred = modelo.predict(df_scaled)\n",
    "    return \"Maligno ⚠️\" if pred[0] == 1 else \"Benigno ✅\", int(pred[0])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cac4916f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
