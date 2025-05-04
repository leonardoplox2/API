import requests

datos = {
    "THINCKNESS": 5,
    "SIZE": 3,
    "SHAPE": 4,
    "ADHESION": 1,
    "SINGLE": 1,
    "NUCLEI": 2,
    "CHROMATIN": 3,
    "NUCLEOLI": 1,
    "MITOSIS": 1
}

r = requests.post("http://127.0.0.1:5000/prediccion", json=datos)
print(r.json())
