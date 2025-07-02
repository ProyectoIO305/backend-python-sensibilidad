from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pulp

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir esto luego
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

    # Crear variables continuas (pulp las trata como tales por defecto)
    x = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(len(coef_objetivo))]

    # Función objetivo
    problema += pulp.lpDot(coef_objetivo, x), "Z"

    # Restricciones
    for i in range(len(lhs)):
        problema += (pulp.lpDot(lhs[i], x) <= rhs[i]), f"R{i+1}"

    # Resolver
    problema.solve()

    if problema.status != 1:  # 1 = Optimal
        return {"mensaje": "No se encontró solución óptima"}

    # Obtener solución óptima
    solucion = {f"x{i+1}": x[i].varValue for i in range(len(x))}
    z_optimo = pulp.value(problema.objective)

    # Analisis de sensibilidad
    sensibilidadVariables = []
    for i, var in enumerate(x):
        sensibilidadVariables.append({
            "variable": var.name,
            "valorActual": round(var.varValue, 4),
            "permisibleAumentar": round(var.dj if var.dj > 0 else 0, 4),
            "permisibleDisminuir": round(-var.dj if var.dj < 0 else 0, 4)
        })

    sensibilidadRestricciones = []
    for nombre, restriccion in problema.constraints.items():
        sombra = restriccion.pi  # Valor sombra
        sensibilidadRestricciones.append({
            "restriccion": nombre,
            "valorActual": round(rhs[int(nombre[1:]) - 1], 4),
            "valorSombra": round(sombra, 4),
            "permisibleAumentar": "No calculado",
            "permisibleDisminuir": "No calculado"
        })

    return {
        "solucion": solucion,
        "z_optimo": round(z_optimo, 4),
        "sensibilidadVariables": sensibilidadVariables,
        "sensibilidadRestricciones": sensibilidadRestricciones
    }