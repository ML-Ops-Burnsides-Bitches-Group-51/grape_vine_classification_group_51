FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi evidently google-cloud-storage --no-cache-dir

COPY src/grape_vine_classification/data_drift_monitoring.py data_drift_monitoring.py

EXPOSE 8007

CMD exec uvicorn data_drift_monitoring:app --port ${PORT:-8007} --host 0.0.0.0