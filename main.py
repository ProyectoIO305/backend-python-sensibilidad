from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

class Restriccion(BaseModel):
  coef: List[float]
  operador: str
  valor: float

class DatosProblema(BaseModel):
  tipo: str
  coefObjetivo: List[float]
  restricciones: List[Restriccion]
  esEntera: List[bool]

@app.get("/")
def inicio():
  return {"mensaje": "Backend Python funcionando 🚀"}

@app.post("/api/sensibilidad")
async def analizar_sensibilidad(datos: DatosProblema):
  print("Datos recibidos:", datos)

  return {
    "mensaje": "Datos recibidos correctamente en Python",
    "tipo": datos.tipo,
    "coefObjetivo": datos.coefObjetivo,
    "restricciones": datos.restricciones,
    "esEntera": datos.esEntera
}
