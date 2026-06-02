FROM python:3.11-slim

WORKDIR /app

# Stable, source-independent dependency layer. Listed explicitly (instead
# of `pip install -e .` against pyproject.toml) so that editing any
# source file under forge/ or scripts/ does NOT bust the deps cache —
# only edits to this Dockerfile line do. Mirrors pyproject.toml exactly
# (no new dependencies introduced here).
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pydantic \
        langchain \
        langchain-groq \
        langchain-community \
        groq \
        ollama \
        psycopg2-binary \
        redis \
        python-dotenv \
        sentence-transformers \
        numpy \
        httpx \
        pytest \
        pytest-asyncio \
        duckduckgo-search \
        slowapi

# Source layer. Anything ignored by .dockerignore (.venv, .git,
# __pycache__, .pytest_cache, *.log) stays out.
COPY . .

# Register the package so `forge.server.main:app` resolves on the
# uvicorn command line. --no-deps because the dependency layer above
# already installed everything; without this flag pip would re-resolve
# the deps and bust the cache.
RUN pip install --no-cache-dir -e . --no-deps

# Force stdout/stderr to be unbuffered so container logs stream in real
# time to `docker compose logs -f api`.
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Production CMD: no --reload (no watchfiles dependency, no inotify
# overhead, no spurious reloads when a mounted volume changes).
CMD ["uvicorn", "forge.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
