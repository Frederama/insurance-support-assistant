## Running with Docker

The application is containerized, so it runs identically on any machine with Docker installed, with no local Python environment required.

```bash
# Build the image
docker build -t insurance-support-assistant .

# Provide your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run
docker compose up
```

The app is then available at `http://localhost:8501`.

### Container design

A few deliberate choices in the Dockerfile:

- **Dependencies are copied and installed before application code.** Docker caches each build step, so placing `COPY requirements.txt` and `pip install` ahead of `COPY . .` means editing application code rebuilds in seconds rather than reinstalling the full dependency tree, which includes PyTorch via sentence-transformers. First build takes roughly seven minutes; subsequent builds take under two seconds.
- **The base image is pinned** to `python:3.11-slim` rather than `latest`, so the build is reproducible over time.
- **The container runs as a non-root user.** Containers run as root by default, which means a compromised process has root inside the container.
- **A healthcheck** polls Streamlit's `/_stcore/health` endpoint, giving an orchestrator a way to know whether the app is serving rather than merely running.
- **`.dockerignore` excludes credentials and local config**, so no secrets are baked into the image.

### Configuration

The API key is resolved from the environment first, falling back to Streamlit secrets:

```python
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None
```

This ordering matters. The original implementation read `st.secrets` first, which raises `StreamlitSecretNotFoundError` when no `secrets.toml` exists anywhere on disk, making the environment-variable fallback unreachable. The bug was invisible on Streamlit Cloud, where a secrets file always exists, and only surfaced when the application was run in a container.

Checking the environment before file-based configuration is the standard convention and means the same image runs locally, on Streamlit Cloud, or anywhere else without modification.
