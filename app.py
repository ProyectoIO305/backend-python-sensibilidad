from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Backend de análisis de sensibilidad funcionando correctamente"}
