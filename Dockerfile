# Imagen base con Python y sistema operativo
FROM python:3.10-slim

# Instalamos glpk (glpsol) usando apt
RUN apt-get update && apt-get install -y glpk-utils

# Crear directorio de trabajo
WORKDIR /app

# Copiar todos los archivos al contenedor
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto (Render usa por defecto el 10000, pero este ajuste es seguro)
EXPOSE 10000

# Comando para ejecutar el backend con uvicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app", "-k", "uvicorn.workers.UvicornWorker"]
