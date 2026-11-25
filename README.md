# Insurance Support Assistant

A multi-agent GenAI system that answers insurance policy questions from real policy documents and customer account questions from a database, routed automatically to the right source.

**[Live Demo ->](https://9ocevdahdtfa5edn6mfnnp.streamlit.app/)**

## Business Problem

Insurance support agents constantly switch between two very different sources: policy documents to answer coverage questions, and customer records to answer account-specific questions. This system unifies both behind a single natural-language interface, automatically routing each question to the right source and combining them when a question needs both.

## Data

- **Policy documents:** 10 real, publicly available policy wordings and endorsements from Axiom Mutual Insurance Company (Ontario), covering Auto, Homeowners (Comprehensive and Named Perils), Tenant, and Condo policies, plus four endorsements (Water Backup, Flood, Transportation Replacement, Depreciation Removal)
- **Customer data:** synthetic, since real customer records are private. 100 customers with policies, claims, and payment history, generated with a fixed random seed for reproducibility

## Architecture

A router classifies each incoming question into one of five paths:

- **POLICY** - answered from policy documents via retrieval
- **SQL** - answered from the customer database via generated SQL
- **HYBRID** - needs both a customer's specific policy and the relevant policy language
- **COMPLAINT** - frustration or urgency with no answerable question, flagged for human follow-up
- **CLARIFY** - not enough information to answer, asks for clarification

Two models are used deliberately: `openai/gpt-oss-20b` (smaller, faster, cheaper) handles routing and SQL generation, since these are simple classification-style tasks. `openai/gpt-oss-120b` handles the actual grounded answer synthesis for POLICY, SQL, and HYBRID responses, where more careful reasoning matters.

## Retrieval: Hybrid Search Plus Guaranteed Sections

Policy documents are chunked and retrieved using a hybrid of dense embeddings (`all-MiniLM-L6-v2` + FAISS) and keyword search (BM25), merged with Reciprocal Rank Fusion. Retrieval is also policy-type-aware: a question naming "auto" or "homeowners" narrows the search to the relevant document, and detected endorsement keywords pull in the matching endorsement document.

**A key finding during development:** neither embeddings nor BM25 could reliably surface certain coverage clauses, even at k=15. The root cause was structural - short, list-style insuring and exclusion clauses (e.g. a policy's core "Insured Perils" provision) don't stand out to either search method next to longer narrative text that happens to repeat the same query words. This is a known limitation of retrieval over dense legal and insurance text, not a bug in the pipeline.

The fix: manually locating and extracting the load-bearing coverage clause from each base policy document (Homeowners "Insured Perils" and Auto "Section 7 - Loss or Damage Coverages") as a guaranteed chunk, force-included whenever that policy type is detected, with hybrid search filling any remaining context. This is a standard pattern in production RAG systems over structured legal documents: guarantee the clauses that matter most, and let search handle everything else.

A related bug was found and fixed during this process: an on-disk index cache was silently reloading a stale, pre-fix version of the chunk list on every notebook restart, discarding the guaranteed-chunk work without any visible error. Deleting the stale cache and rebuilding resolved it.

## SQL Agent

Customer questions are converted to SQL by an LLM constrained to the database schema, SELECT-only, with explicit rules to prevent two real failure modes found during testing:

- Partial or first-name-only questions ("Does Sarah have flood coverage?") now match with `LIKE` instead of requiring an exact name
- The model does not conflate endorsements (flood, water backup) with the `policies.type` column, which only contains actual policy types - an earlier version attempted to filter policies by `type = 'flood'`, which can never match anything

## Evaluation

A 13-question evaluation set covering all five routes was used throughout development. Final router accuracy: **13/13**.

## Key Design Decisions

- **Partial name matching throughout** - both the SQL agent and the hybrid agent's customer lookup use `LIKE` rather than exact match, since a real support agent will often have only a first name or partial spelling
- **Chunk truncation for cost control** - guaranteed chunks are 12,000-19,000 characters; only the first 1,200 characters of each retrieved chunk are sent to the LLM, since that reliably captures the operative language without paying for the full section on every call
- **Model tiering** - simple classification tasks use the smaller, cheaper model; grounded answer generation uses the larger one

## Limitations

- Customer data is synthetic; real deployment would connect to an actual policy administration system
- A question giving only a first name with no other identifying detail and multiple same-name matches will return the first match rather than asking which customer is meant
- The guaranteed-chunk approach is currently manual per document; scaling to a full policy library would need a more systematic way to identify and extract core coverage clauses
- This is a support-assistant prototype, not a source of insurance advice; coverage answers should be confirmed against the customer's actual Declaration Page and policy documents

## Project Structure

```
insurance-support-assistant/
├── app.py                  # Streamlit app - full agent pipeline
├── requirements.txt
├── Dockerfile               # Container definition
├── docker-compose.yml       # Local run with environment config
├── .dockerignore            # Excludes secrets and local config from the image
├── insurance.db             # Synthetic customer database
├── policy_index.faiss       # FAISS embedding index
├── policy_chunks.pkl        # Chunked policy text with metadata
└── .streamlit/
    └── secrets.toml         # Groq API key (not committed)
```

## How to Run Locally

```bash
git clone https://github.com/Frederama/insurance-support-assistant.git
cd insurance-support-assistant
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-key-here"
```

```bash
streamlit run app.py
```

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

## Tech Stack

- **Retrieval:** sentence-transformers, FAISS, rank_bm25
- **LLM:** Groq (GPT-OSS 120B and 20B)
- **App:** Streamlit
- **Data:** SQLite, pypdf
- **Deployment:** Docker, Streamlit Cloud

## Credit

Policy documents are publicly available wordings from Axiom Mutual Insurance Company. This project is not affiliated with or endorsed by Axiom Mutual, and nothing here should be treated as insurance advice.
