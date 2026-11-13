# UNTESTED scaffold for Cloud Run (could not be built in the authoring sandbox).
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY invest_advisor ./agents/invest_advisor
COPY data ./agents/data
# Cloud Run injects PORT; ADK serves the REST API (agents dir contains the agent package).
# Cloud Run's filesystem is read-only except /tmp, so disable the local audit file: the
# stdout JSON line is ingested by Cloud Logging automatically.
ENV LOCAL_AUDIT_LOG=false
CMD ["sh", "-c", "adk api_server --host 0.0.0.0 --port ${PORT:-8080} ./agents"]
