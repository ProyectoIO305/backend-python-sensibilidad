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

    sensibilidadVariables = []

    # Sensibilidad aproximada numérica
    for i, var in enumerate(x):
        if coef_objetivo[i] == 0:
            sensibilidadVariables.append({
                "variable": var.name,
                "valorActual": round(var.varValue, 4),
                "variacionAproximada": "±0%"
            })
            continue

        # Aumentar 10%
        coef_aumentado = coef_objetivo.copy()
        coef_aumentado[i] *= 1.1

        problema_temp = pulp.LpProblem("Temp", pulp.LpMaximize if tipo == 'max' else pulp.LpMinimize)
        x_temp = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(len(coef_objetivo))]
        problema_temp += pulp.lpDot(coef_aumentado, x_temp), "Z"

        for j in range(len(lhs)):
            problema_temp += (pulp.lpDot(lhs[j], x_temp) <= rhs[j]), f"R{j+1}"

        problema_temp.solve()
        z_aumentado = pulp.value(problema_temp.objective)

        # Disminuir 10%
        coef_disminuido = coef_objetivo.copy()
        coef_disminuido[i] *= 0.9

        problema_temp = pulp.LpProblem("Temp", pulp.LpMaximize if tipo == 'max' else pulp.LpMinimize)
        x_temp = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(len(coef_objetivo))]
        problema_temp += pulp.lpDot(coef_disminuido, x_temp), "Z"

        for j in range(len(lhs)):
            problema_temp += (pulp.lpDot(lhs[j], x_temp) <= rhs[j]), f"R{j+1}"

        problema_temp.solve()
        z_disminuido = pulp.value(problema_temp.objective)

        # Calcular el porcentaje de variación de Z
        variacion_mas = abs((z_aumentado - z_optimo) / z_optimo) * 100 if z_optimo != 0 else 0
        variacion_menos = abs((z_disminuido - z_optimo) / z_optimo) * 100 if z_optimo != 0 else 0

        sensibilidadVariables.append({
            "variable": var.name,
            "valorActual": round(var.varValue, 4),
            "variacionAproximada": f"±{round(max(variacion_mas, variacion_menos), 2)}%"
        })

    return {
        "solucion": solucion,
        "z_optimo": round(z_optimo, 4),
        "sensibilidadVariables": sensibilidadVariables
    }
