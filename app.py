from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import scipy.optimize as opt

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pruebaproyecto-1jeh.onrender.com"],  # Específico
    allow_credentials=True,
    allow_methods=["*"],  # Puedes usar ["GET", "POST", "OPTIONS"] pero "*" es más flexible
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend de análisis de sensibilidad funcionando correctamente"}

@app.options("/analisis-sensibilidad")  # Soportar preflight OPTIONS
def options_handler():
    return {}

@app.post("/analisis-sensibilidad")
async def analisis_sensibilidad(request: Request):
    try:
        body = await request.json()
        print("🔵 Datos recibidos:", body)

        tipo = body["tipo"]
        coef_objetivo = np.array(body["coef_objetivo"])
        lhs = np.array(body["lhs"])
        rhs = np.array(body["rhs"])

        print("✅ Datos procesados correctamente.")
        print(f"tipo: {tipo}")
        print(f"coef_objetivo: {coef_objetivo}")
        print(f"lhs: {lhs}")
        print(f"rhs: {rhs}")

        c = coef_objetivo if tipo == "min" else -coef_objetivo

        resultado = opt.linprog(c=c, A_ub=lhs, b_ub=rhs, method="highs")

        if not resultado.success:
            return {"success": False, "message": "No se pudo resolver el problema."}

        solucion = resultado.x
        valor_objetivo = resultado.fun if tipo == "min" else -resultado.fun

        sensibilidad_variables = [
            {"variable": f"x{i+1}", "valorActual": v, "permisibleAumentar": "N/A", "permisibleDisminuir": "N/A"}
            for i, v in enumerate(solucion)
        ]

        sensibilidad_restricciones = [
            {"restriccion": f"Restricción {i+1}", "valorActual": rhs[i], "valorSombra": "N/A", "permisibleAumentar": "N/A", "permisibleDisminuir": "N/A"}
            for i in range(len(rhs))
        ]

        return {
            "success": True,
            "solucion": solucion.tolist(),
            "valor_objetivo": valor_objetivo,
            "sensibilidadVariables": sensibilidad_variables,
            "sensibilidadRestricciones": sensibilidad_restricciones
        }

    except Exception as e:
        print(f"❌ Error procesando la solicitud: {str(e)}")
        return {"success": False, "message": f"Error en el servidor: {str(e)}"}

