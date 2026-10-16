FROM python:3.11-slim

ARG APP_VERSION=dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_VERSION=${APP_VERSION}

WORKDIR /app

COPY requirements/core.txt requirements/api.txt ./requirements/
RUN pip install --no-cache-dir -r requirements/core.txt -r requirements/api.txt

COPY api ./api
COPY worker ./worker
COPY agent ./agent
COPY core ./core
COPY migrations ./migrations

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
