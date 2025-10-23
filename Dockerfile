FROM python:3.10-slim

ENV PIP_NO_INPUT=1 \
    PIPENV_VENV_IN_PROJECT=1 \
    OLLAMA_HOST=http://host.docker.internal:11434

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    pip install --no-cache-dir pipenv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Pipfile* ./
RUN pipenv install --deploy --dev --system
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python manage.py migrate && python -u manage.py runserver 0.0.0.0:8000"]