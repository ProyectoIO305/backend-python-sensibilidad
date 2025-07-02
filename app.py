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
    tipo: str
    coef_objetivo: list
    lhs: list
    rhs: list

@app.post("/analisis-sensibilidad")
async def analisis_sensibilidad(data: SensibilidadRequest):
    tipo = data.tipo
    coef_objetivo = data.coef_objetivo
    lhs = data.lhs
    rhs = data.rhs

    problema = pulp.LpProblem("PL_Sensibilidad", pulp.LpMaximize if tipo == 'max' else pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(len(coef_objetivo))]

    problema += pulp.lpDot(coef_objetivo, x), "Z"

    for i in range(len(lhs)):
        problema += (pulp.lpDot(lhs[i], x) <= rhs[i]), f"R{i+1}"

    problema.solve()

    if problema.status != 1:
        return {"mensaje": "No se encontró solución óptima"}

    solucion = {f"x{i+1}": x[i].varValue for i in range(len(x))}
    z_optimo = pulp.value(problema.objective)

    # Sensibilidad de variables
    sensibilidadVariables = []
    for i, var in enumerate(x):
        if coef_objetivo[i] != 0:
            porcentaje = abs(var.dj / coef_objetivo[i]) * 100
        else:
            porcentaje = 0

        sensibilidadVariables.append({
            "variable": var.name,
            "valorActual": round(var.varValue, 4),
            "variacionAproximada": f"±{round(porcentaje, 2)}%"
        })

    # Sensibilidad de restricciones
    sensibilidadRestricciones = []
    for nombre, restriccion in problema.constraints.items():
        sombra = restriccion.pi
        valor_actual = rhs[int(nombre[1:]) - 1]

        if sombra != 0:
            porcentaje = abs(valor_actual / sombra) * 100
        else:
            porcentaje = 0

        sensibilidadRestricciones.append({
            "restriccion": nombre,
            "valorActual": round(valor_actual, 4),
            "valorSombra": round(sombra, 4),
            "variacionAproximada": f"±{round(porcentaje, 2)}%"
        })

    return {
        "solucion": solucion,
        "z_optimo": round(z_optimo, 4),
        "sensibilidadVariables": sensibilidadVariables,
        "sensibilidadRestricciones": sensibilidadRestricciones
    }
