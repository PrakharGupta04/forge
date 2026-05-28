FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install -e . --no-cache-dir

COPY . .

CMD ["uvicorn", "forge.server.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
