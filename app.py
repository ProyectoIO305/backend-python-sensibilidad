from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pulp
import xmlrpc.client
import time

app = FastAPI()

@app.get("/")
def read_root():
    return {"mensaje": "El backend está funcionando correctamente"}

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

    # Resolver localmente con pulp (por si quieres compararlo)
    problema.solve()

    if problema.status != 1:  # 1 = Optimal
        return {"mensaje": "No se encontró solución óptima"}

    # Solución local
    solucion_local = {f"x{i+1}": x[i].varValue for i in range(len(x))}
    z_optimo_local = pulp.value(problema.objective)

    # Conectar a NEOS
    mps_model = generar_mps(coef_objetivo, lhs, rhs, tipo)
    sensibilidad_neos = enviar_a_neos(mps_model)

    return {
        "solucion_local": solucion_local,
        "z_optimo_local": round(z_optimo_local, 4),
        "respuesta_neos": sensibilidad_neos
    }

def generar_mps(coef_objetivo, lhs, rhs, tipo):
    """
    Genera un modelo MPS en formato texto desde los datos JSON.
    """
    n_vars = len(coef_objetivo)
    n_restricciones = len(lhs)
    
    lines = []
    lines.append("NAME          PL_SENSIBILIDAD")
    lines.append("ROWS")
    lines.append(" N  OBJETIVO")
    for i in range(n_restricciones):
        lines.append(f" L  R{i+1}")

    lines.append("COLUMNS")
    for j in range(n_vars):
        var_name = f"x{j+1}"
        lines.append(f"    {var_name}  OBJETIVO    {coef_objetivo[j]}")
        for i in range(n_restricciones):
            if lhs[i][j] != 0:
                lines.append(f"    {var_name}  R{i+1}    {lhs[i][j]}")

    lines.append("RHS")
    for i in range(n_restricciones):
        lines.append(f"    RHS1      R{i+1}    {rhs[i]}")

    lines.append("BOUNDS")
    for j in range(n_vars):
        lines.append(f" LO BOUND     x{j+1}    0")

    lines.append("ENDATA")

    return "\n".join(lines)

def enviar_a_neos(mps_model):
    neos = xmlrpc.client.ServerProxy("https://neos-server.org:3333")
    
    solver = "lp"
    solver_category = "lp"
    solver_name = "CPLEX"

    # Crear un trabajo XML
    mps_model = mps_model.replace("&", "&amp;")  # Para evitar errores XML
    input_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
    <category>{solver_category}</category>
    <solver>{solver_name}</solver>
    <inputMethod>MPS</inputMethod>
    <inputType>AMPL</inputType>
    <MPSInput><![CDATA[
{mps_model}
    ]]></MPSInput>
</document>
"""

    # Enviar el trabajo a NEOS
    job_number, password = neos.submitJob(input_xml)
    print(f"Enviado a NEOS. Job#: {job_number}, Password: {password}")

    # Esperar a que el trabajo se procese
    status = ""
    while status != "Done":
        time.sleep(3)
        status = neos.getJobStatus(job_number, password)
        print(f"Estado del trabajo NEOS: {status}")

    # Obtener resultados
    result = neos.getFinalResults(job_number, password)
    return {"resultado_raw": result.decode("utf-8")}
