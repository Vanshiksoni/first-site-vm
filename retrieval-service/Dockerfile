FROM python:3.12-slim

WORKDIR /app

COPY app_docker.py app.py
COPY static ./static

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
