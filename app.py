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

    # Sensibilidad de variables (aproximada)
    sensibilidadVariables = []
    for i, var in enumerate(x):
        if coef_objetivo[i] == 0:
            sensibilidadVariables.append({
                "variable": var.name,
                "valorActual": round(var.varValue, 4),
                "comentario": "El coeficiente actual es 0, no afecta directamente a Z"
            })
            continue

        # Aumentar 10%
        coef_aumentado = coef_objetivo.copy()
        coef_aumentado[i] *= 1.1

        problema_temp = pulp.LpProblem("TempAumento", pulp.LpMaximize if tipo == 'max' else pulp.LpMinimize)
        x_temp = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(len(coef_objetivo))]
        problema_temp += pulp.lpDot(coef_aumentado, x_temp), "Z"

        for j in range(len(lhs)):
            problema_temp += (pulp.lpDot(lhs[j], x_temp) <= rhs[j]), f"R{j+1}"

        problema_temp.solve()
        z_mayor = pulp.value(problema_temp.objective)

        # Disminuir 10%
        coef_disminuido = coef_objetivo.copy()
        coef_disminuido[i] *= 0.9

        problema_temp = pulp.LpProblem("TempDisminucion", pulp.LpMaximize if tipo == 'max' else pulp.LpMinimize)
        x_temp = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(len(coef_objetivo))]
        problema_temp += pulp.lpDot(coef_disminuido, x_temp), "Z"

        for j in range(len(lhs)):
            problema_temp += (pulp.lpDot(lhs[j], x_temp) <= rhs[j]), f"R{j+1}"

        problema_temp.solve()
        z_menor = pulp.value(problema_temp.objective)

        comentario = f"Si el coeficiente de {var.name} varía ±10%, Z estaría entre {round(z_menor, 2)} y {round(z_mayor, 2)}."

        sensibilidadVariables.append({
            "variable": var.name,
            "valorActual": round(var.varValue, 4),
            "comentario": comentario
        })

    # Sensibilidad de restricciones (aproximada)
    sensibilidadRestricciones = []
    for nombre, restriccion in problema.constraints.items():
        sombra = restriccion.pi if restriccion.pi is not None else 0
        index = int(nombre[1:]) - 1
        valor_actual = rhs[index]

        # Aumentar 10% el lado derecho
        rhs_aumentado = rhs.copy()
        rhs_aumentado[index] *= 1.1

        problema_temp = pulp.LpProblem("TempRestriccionAumento", pulp.LpMaximize if tipo == 'max' else pulp.LpMinimize)
        x_temp = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(len(coef_objetivo))]
        problema_temp += pulp.lpDot(coef_objetivo, x_temp), "Z"

        for j in range(len(lhs)):
            problema_temp += (pulp.lpDot(lhs[j], x_temp) <= rhs_aumentado[j]), f"R{j+1}"

        problema_temp.solve()
        z_mayor = pulp.value(problema_temp.objective)

        # Disminuir 10% el lado derecho
        rhs_disminuido = rhs.copy()
        rhs_disminuido[index] *= 0.9

        problema_temp = pulp.LpProblem("TempRestriccionDisminucion", pulp.LpMaximize if tipo == 'max' else pulp.LpMinimize)
        x_temp = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(len(coef_objetivo))]
        problema_temp += pulp.lpDot(coef_objetivo, x_temp), "Z"

        for j in range(len(lhs)):
            problema_temp += (pulp.lpDot(lhs[j], x_temp) <= rhs_disminuido[j]), f"R{j+1}"

        problema_temp.solve()
        z_menor = pulp.value(problema_temp.objective)

        comentario = f"Si el lado derecho de {nombre} varía ±10%, Z estaría entre {round(z_menor, 2)} y {round(z_mayor, 2)}."

        sensibilidadRestricciones.append({
            "restriccion": nombre,
            "valorActual": round(valor_actual, 4),
            "valorSombra": round(sombra, 4),
            "comentario": comentario
        })

    return {
        "solucion": solucion,
        "z_optimo": round(z_optimo, 4),
        "sensibilidadVariables": sensibilidadVariables,
        "sensibilidadRestricciones": sensibilidadRestricciones
    }
