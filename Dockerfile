# Insurance Support Assistant - multi-agent RAG system
# Base pinned to a specific minor version for reproducible builds
FROM python:3.11-slim

# Avoid .pyc files and force unbuffered logs so container output streams correctly
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies before copying source so this layer is cached
# across rebuilds when only application code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data artifacts
COPY . .

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit needs to bind to all interfaces to be reachable from outside the container
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py"]
