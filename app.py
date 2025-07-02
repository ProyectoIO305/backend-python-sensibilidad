from fastapi import FastAPI, Request
import numpy as np
import scipy.optimize as opt

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Backend de análisis de sensibilidad funcionando correctamente"}

@app.post("/analisis-sensibilidad")
async def analisis_sensibilidad(request: Request):
    body = await request.json()
    tipo = body["tipo"]
    coef_objetivo = np.array(body["coef_objetivo"])
    lhs = np.array(body["lhs"])
    rhs = np.array(body["rhs"])

    c = coef_objetivo if tipo == "min" else -coef_objetivo

    resultado = opt.linprog(c=c, A_ub=lhs, b_ub=rhs, method="highs")

    if not resultado.success:
        return {"success": False, "message": "No se pudo resolver el problema."}

    solucion = resultado.x
    valor_objetivo = resultado.fun if tipo == "min" else -resultado.fun

    return {
        "success": True,
        "solucion": solucion.tolist(),
        "valor_objetivo": valor_objetivo
    }
