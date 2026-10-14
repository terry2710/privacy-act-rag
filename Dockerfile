FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-core.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt -r requirements-api.txt

COPY api ./api
COPY worker ./worker
COPY agent ./agent
COPY rag_backend.py rag_logging.py rag_retrieval.py rag_evaluation.py ./
COPY data ./data

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
