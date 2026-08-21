FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[integrations]"

COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY .env.example .env.example

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
