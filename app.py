from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pulp

app = FastAPI()

@app.get("/")
def read_root():
    return {"mensaje": "El backend está funcionando correctamente"}

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Modelo de datos esperado
class SensibilidadRequest(BaseModel):
    tipo: str  # 'max' o 'min'
    coef_objetivo: list
    lhs: list  # Lista de listas (coeficientes restricciones)
    rhs: list  # Lado derecho restricciones

@app.post("/analisis-sensibilidad")
async def analisis_sensibilidad(data: SensibilidadRequest):
    tipo = data.tipo
    coef_objetivo = data.coef_objetivo
    lhs = data.lhs
    rhs = data.rhs

    # Crear problema
    problema = pulp.LpProblem("PL_Sensibilidad", pulp.LpMaximize if tipo == 'max' else pulp.LpMinimize)

    # Crear variables continuas
    x = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(len(coef_objetivo))]

    # Función objetivo
    problema += pulp.lpDot(coef_objetivo, x), "Z"

    # Restricciones
    for i in range(len(lhs)):
        problema += (pulp.lpDot(lhs[i], x) <= rhs[i]), f"R{i+1}"

    # Resolver
    problema.solve()

    if problema.status != 1:
        return {"mensaje": "No se encontró solución óptima"}

    solucion = {f"x{i+1}": x[i].varValue for i in range(len(x))}
    z_optimo = pulp.value(problema.objective)

    sensibilidadVariables = []
    for i, var in enumerate(x):
        if var.varValue == 0:
            comentario = "Variable no utilizada en la solución actual"
        else:
            comentario = "Variable activa en la solución actual"

        sensibilidadVariables.append({
            "variable": var.name,
            "valorActual": round(var.varValue, 4),
            "comentario": comentario
        })

    sensibilidadRestricciones = []
    for nombre, restriccion in problema.constraints.items():
        sombra = restriccion.pi
        sensibilidadRestricciones.append({
            "restriccion": nombre,
            "valorActual": round(rhs[int(nombre[1:]) - 1], 4),
            "valorSombra": round(sombra, 4),
            "comentario": "Análisis detallado no disponible"
        })

    return {
        "solucion": solucion,
        "z_optimo": round(z_optimo, 4),
        "sensibilidadVariables": sensibilidadVariables,
        "sensibilidadRestricciones": sensibilidadRestricciones
    }
