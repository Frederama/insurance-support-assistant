import streamlit as st
import sqlite3
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Support Assistant",
    page_icon="🛡️",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.main, .stApp { background-color: #f7f6f2; }

.hero {
    background: #0f1923;
    color: #f7f6f2;
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2.2rem; margin-bottom: 0.3rem; color: #f7f6f2; }
.hero p  { color: #8a9ba8; font-size: 1rem; font-weight: 300; margin: 0; }

.route-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.badge-POLICY    { background: #dbeafe; color: #1d4ed8; }
.badge-SQL       { background: #dcfce7; color: #15803d; }
.badge-HYBRID    { background: #ede9fe; color: #6d28d9; }
.badge-COMPLAINT { background: #fee2e2; color: #b91c1c; }
.badge-CLARIFY   { background: #fef3c7; color: #a16207; }

.answer-box {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.75rem 2rem;
    border: 1px solid #e5e7eb;
    line-height: 1.6;
    color: #1f2937;
}

.sources-box {
    background: #f9fafb;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    font-size: 0.8rem;
    color: #6b7280;
}

.section-label {
    font-size: 0.75rem; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b7280;
    margin-bottom: 0.5rem; margin-top: 1.25rem;
}

div[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #0f1923 !important;
}

.stButton > button {
    background-color: #0f1923; color: #f7f6f2;
    border: none; border-radius: 8px;
    padding: 0.65rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500; font-size: 0.95rem;
    letter-spacing: 0.03em;
}
.stButton > button:hover { background-color: #1e3a4f; }

.example-chip {
    display: inline-block;
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.78rem;
    color: #374151;
    margin: 0.2rem 0.3rem 0.2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Schema (for SQL agent) ───────────────────────────────────────────────────────
SCHEMA = """
customers(
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    phone TEXT,
    city TEXT,
    province TEXT,
    policy_id INTEGER
)

policies(
    policy_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    type TEXT,
    coverage_amount REAL,
    deductible REAL,
    monthly_premium REAL,
    start_date TEXT,
    end_date TEXT,
    status TEXT
)

claims(
    claim_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    policy_id INTEGER,
    claim_type TEXT,
    amount REAL,
    status TEXT,
    date_filed TEXT,
    description TEXT
)

payments(
    payment_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount REAL,
    due_date TEXT,
    status TEXT
)
"""

# ── Policy document maps ─────────────────────────────────────────────────────────
POLICY_DOCUMENT_MAP = {
    "Homeowners Named Perils": "PP-1701-0125-Residential-Homeowners-Named-Perils-Policy.pdf",
    "Homeowners Comprehensive": "PP-1703-0125-Residential-Homeowners-Comprehensive-Policy.pdf",
    "Tenant Comprehensive": "PP-1055-0125-Tenant-Comprehensive-Policy.pdf",
    "Condo Unit Owners Comprehensive": "PP-1705-0125-Residential-Condo-Unit-Owners-Comprehensive-Policy.pdf",
    "Auto": "OAP1-2022_aoda.pdf"
}

ENDORSEMENT_MAP = {
    "water": ["EO-1025-0124-Water-Backup-Coverage-Endorsement.pdf"],
    "flood": ["EO-1015-0915-Flood-Coverage-Endorsement.pdf"],
    "transportation": ["OPCF-20-Coverage-for-Transportation-Replacement.pdf"],
    "depreciation": ["OPCF-43A-Removing-Depreciation-Deduction-for-Specified-Lessees.pdf"]
}


def detect_policy_type(query):
    q = query.lower()
    if "auto" in q or "car" in q or "vehicle" in q:
        return "Auto"
    if "tenant" in q or "renter" in q:
        return "Tenant Comprehensive"
    if "condo" in q:
        return "Condo Unit Owners Comprehensive"
    if "named perils" in q:
        return "Homeowners Named Perils"
    if "homeowners" in q or "homeowner" in q:
        return "Homeowners Comprehensive"
    return None


def detect_endorsements(query):
    q = query.lower()
    detected = []
    for keyword, files in ENDORSEMENT_MAP.items():
        if keyword in q:
            detected.extend(files)
    return detected


# ── Load artifacts (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    index = faiss.read_index("policy_index.faiss")
    with open("policy_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    tokenized_chunks = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, bm25, embedding_model


@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        import os
        api_key = os.environ.get("GROQ_API_KEY")
    return Groq(api_key=api_key)


index, chunks, bm25, embedding_model = load_artifacts()
client = get_groq_client()

MODEL_FAST = "openai/gpt-oss-20b"
MODEL_MAIN = "openai/gpt-oss-120b"
DB_PATH = "insurance.db"


# ── Retrieval ──────────────────────────────────────────────────────────────────
def hybrid_retrieve(query, k=5, rrf_k=60):
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    _, faiss_indices = index.search(query_embedding, 20)
    faiss_ranks = {idx: rank for rank, idx in enumerate(faiss_indices[0])}

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_top)}

    all_indices = set(faiss_ranks) | set(bm25_ranks)
    fused_scores = {}
    for idx in all_indices:
        score = 0
        if idx in faiss_ranks:
            score += 1 / (rrf_k + faiss_ranks[idx])
        if idx in bm25_ranks:
            score += 1 / (rrf_k + bm25_ranks[idx])
        fused_scores[idx] = score

    top_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]
    return [{"score": fused_scores[i], "source": chunks[i]["source"], "text": chunks[i]["text"]} for i in top_indices]


def retrieve_policy_chunks(query, k=5):
    policy_type = detect_policy_type(query)
    endorsement_files = detect_endorsements(query)

    forced_chunks = []
    if policy_type:
        target_file = POLICY_DOCUMENT_MAP.get(policy_type)
        forced_chunks.extend([
            {"score": 1.0, "source": c["source"], "text": c["text"]}
            for c in chunks if c.get("guaranteed") and c["source"] == target_file
        ])

    forced_texts = {c["text"] for c in forced_chunks}
    remaining_k = max(k - len(forced_chunks), 1)

    hybrid_results = hybrid_retrieve(query, k=remaining_k + 3)
    for r in hybrid_results:
        if r["text"] not in forced_texts and len(forced_chunks) < k:
            forced_chunks.append(r)

    return forced_chunks[:k]


def retrieve_hybrid_policy(query, policy_type, k=3):
    policy_file = POLICY_DOCUMENT_MAP.get(policy_type)
    if not policy_file:
        return []

    relevant_files = [policy_file]
    q = query.lower()
    for keyword, files in ENDORSEMENT_MAP.items():
        if keyword in q:
            relevant_files.extend(files)

    relevant_chunks = [c for c in chunks if c["source"] in relevant_files]
    if not relevant_chunks:
        return []

    relevant_embeddings = embedding_model.encode([c["text"] for c in relevant_chunks])
    faiss.normalize_L2(relevant_embeddings)

    mini_index = faiss.IndexFlatIP(relevant_embeddings.shape[1])
    mini_index.add(relevant_embeddings)

    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)

    scores, indices = mini_index.search(query_embedding, min(k, len(relevant_chunks)))

    return [
        {"score": float(score), "source": relevant_chunks[idx]["source"], "text": relevant_chunks[idx]["text"]}
        for score, idx in zip(scores[0], indices[0])
    ]


# ── Agents ─────────────────────────────────────────────────────────────────────
def policy_agent(query, k=3, max_chars_per_chunk=1200):
    results = retrieve_policy_chunks(query, k)

    context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text'][:max_chars_per_chunk]}"
        for r in results
    )

    prompt = f"""
You are an insurance policy support assistant.

Answer the customer's question ONLY using the policy excerpts provided below.

Rules:
- Do not use outside insurance knowledge.
- Do not assume coverage unless the excerpts explicitly support it.
- Do not infer that a section is a coverage provision based only on related wording.
- Preserve the meaning and structure of the policy. Do not relabel exclusions as covered perils or covered perils as exclusions.
- If the excerpts contain an exclusion, describe it as an exclusion.
- If the excerpts contain a condition for coverage, describe it as a condition.
- Do not claim that a policy contains or excludes something unless the provided excerpts explicitly establish it.
- Do not infer coverage from another policy type.
- If an endorsement or Declaration Page condition is mentioned, clearly state that the customer's actual attachment cannot be confirmed unless the provided evidence shows it.
- Distinguish between what the excerpts establish and what cannot be determined.
- Do not invent examples, exclusions, limits, deductibles, or conditions.
- Keep the answer concise and easy to understand.
- Identify the source document when referring to a specific provision.

Customer question:
{query}

Policy excerpts:
{context}
"""

    response = client.chat.completions.create(
        model=MODEL_MAIN,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    sources = list(dict.fromkeys(r["source"] for r in results))
    return {"answer": response.choices[0].message.content, "sources": sources}


def generate_sql(query):
    prompt = f"""
You are an SQL assistant for an insurance support system.

Convert the customer's question into a SQLite SQL query using ONLY the schema provided.

Rules:
- Return ONLY the SQL query.
- Do not use markdown or code fences.
- Do not modify the database.
- Only use SELECT statements.
- Use exact table and column names from the schema.
- Join tables when necessary.
- Text comparisons must be case-insensitive. Use LOWER() when comparing text values.
- When the customer question gives only a first name or a partial name, match using LOWER(name) LIKE LOWER('%partial%') instead of an exact match.
- The policies table's "type" column only contains: Auto, Homeowners Named Perils, Homeowners Comprehensive, Tenant Comprehensive, Condo Unit Owners Comprehensive. Never filter policies.type by coverage topics like "flood" or "water damage" - those are endorsements, not policy types, and endorsement attachment is not stored in this database.
- If the question asks about coverage for a specific peril or endorsement (e.g. flood, water backup), retrieve only the customer's policy type and status - do not attempt to filter or search for the peril itself in SQL.
- If the question cannot be answered from the schema, return: CANNOT_ANSWER
- Always SELECT the customer's name alongside any claim, payment, or policy details being asked about, so the answer can reference who the result belongs to.

Database schema:
{SCHEMA}

Customer question:
{query}
"""
    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def execute_sql(sql):
    if not sql.strip().lower().startswith("select"):
        return "Only SELECT queries are allowed."
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description]
        return columns, rows
    except sqlite3.Error as e:
        return f"SQL error: {e}"
    finally:
        conn.close()


def sql_agent(query):
    sql = generate_sql(query)
    if sql == "CANNOT_ANSWER":
        return "I cannot answer that question using the available customer database."

    result = execute_sql(sql)
    if isinstance(result, str):
        return result

    columns, rows = result
    prompt = f"""
You are an insurance customer support assistant.

Answer the customer's question using the database result below.

Rules:
- Use only the information in the database result.
- Do not invent information.
- Give a clear and concise answer.
- Do not mention SQL, databases, queries, or internal system details.

Customer question:
{query}

Database result:
Columns: {columns}
Rows: {rows}
"""
    response = client.chat.completions.create(
        model=MODEL_MAIN,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def get_customer_policy_info(customer_name):
    sql = f"""
    SELECT c.customer_id, c.name, p.policy_id, p.type, p.coverage_amount,
           p.deductible, p.monthly_premium, p.start_date, p.end_date, p.status
    FROM customers c
    JOIN policies p ON c.policy_id = p.policy_id
    WHERE LOWER(c.name) LIKE LOWER('%{customer_name}%')
    """
    result = execute_sql(sql)
    if isinstance(result, str):
        return None
    columns, rows = result
    return {"columns": columns, "rows": rows}


def hybrid_agent(customer_name, query, max_chars_per_chunk=1200):
    customer_info = get_customer_policy_info(customer_name)
    if not customer_info or not customer_info["rows"]:
        return "I could not find that customer in the available database."

    policy_type = customer_info["rows"][0][3]
    policy_results = retrieve_hybrid_policy(query, policy_type)

    context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text'][:max_chars_per_chunk]}"
        for r in policy_results
    )
    customer_context = f"Columns: {customer_info['columns']}\nRows: {customer_info['rows']}"

    prompt = f"""
You are an insurance customer support assistant.

Answer the customer's question using ONLY the customer information
and policy excerpts provided below.

Rules:
- Do not use outside insurance knowledge.
- Do not assume coverage exists unless the provided evidence supports it.
- Do not claim that a policy does or does not contain a provision unless
  the provided excerpts explicitly establish that.
- Distinguish between the customer's actual policy information and
  general policy provisions.
- If an endorsement or condition must be confirmed from the customer's
  policy or Declaration Page, clearly say that it cannot be confirmed
  from the available information.
- Do not invent coverage, exclusions, limits, deductibles, or conditions.
- Keep the answer concise and easy to understand.
- Identify the relevant policy document when referring to a provision.

Customer information:
{customer_context}

Policy excerpts:
{context}

Customer question:
{query}
"""
    response = client.chat.completions.create(
        model=MODEL_MAIN,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def route_query(query):
    prompt = f"""
You are a query router for an insurance support system.

Choose exactly one route:

POLICY - questions that can be answered using insurance policy documents,
including coverage, exclusions, definitions, perils, and endorsements.

SQL - questions about specific customer information stored in the database,
including policies, claims, payments, premiums, contact information,
locations, and policy status.

HYBRID - questions that require BOTH customer-specific information and
policy-document information to answer.

COMPLAINT - messages expressing frustration, dissatisfaction, or urgency
about service quality, wait times, or being unhelped, where no specific
factual question is being asked that POLICY, SQL, or HYBRID could answer.

CLARIFY - questions that cannot be answered without additional customer
information, clarification, or information that is not available in the
system.

Examples:
"What does the Water Backup Endorsement cover?" -> POLICY
"What is Gina Carter's monthly premium?" -> SQL
"Does Gina Carter's policy cover flood damage?" -> HYBRID
"I've been waiting 3 months and nobody is helping me!" -> COMPLAINT
"What is the best insurance policy for me?" -> CLARIFY
"Can I increase my coverage?" -> CLARIFY

Return ONLY one word:
POLICY
SQL
HYBRID
COMPLAINT
or
CLARIFY

Customer question:
{query}
"""
    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    route = response.choices[0].message.content.strip().upper()
    if "COMPLAINT" in route:
        return "COMPLAINT"
    if "CLARIFY" in route:
        return "CLARIFY"
    if "HYBRID" in route:
        return "HYBRID"
    if "POLICY" in route:
        return "POLICY"
    return "SQL"


def support_agent_final(query, customer_name=None):
    route = route_query(query)

    if route == "POLICY":
        result = policy_agent(query)
        answer, sources = result["answer"], result["sources"]
    elif route == "SQL":
        answer, sources = sql_agent(query), []
    elif route == "HYBRID":
        if not customer_name:
            answer, sources = "I need the customer's name to answer this question.", []
        else:
            answer, sources = hybrid_agent(customer_name, query), []
    elif route == "COMPLAINT":
        answer, sources = (
            "I'm sorry to hear about your experience. I've flagged this for "
            "priority review by a support representative who will follow up "
            "with you directly."
        ), []
    else:
        answer, sources = "I need more information to determine how to answer this question.", []

    return {"route": route, "answer": answer, "sources": sources}


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Insurance Support Assistant</h1>
    <p>A multi-agent system that answers policy questions from real insurance documents
       and customer questions from account data - routed automatically to the right source.</p>
</div>
""", unsafe_allow_html=True)

col_input, col_result = st.columns([1, 1.4], gap="large")

with col_input:
    st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)
    query = st.text_input("Question", placeholder="e.g. Does the homeowners policy cover fire damage?", label_visibility="collapsed")

    st.markdown('<div class="section-label">Customer Name (only needed for account-specific questions)</div>', unsafe_allow_html=True)
    customer_name = st.text_input("Customer name", placeholder="e.g. Gina Carter", label_visibility="collapsed")

    ask_btn = st.button("Ask")

    st.markdown('<div class="section-label">Try an Example</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="example-chip">Does the homeowners policy cover fire damage?</span>
    <span class="example-chip">What does the Water Backup Endorsement cover?</span>
    <span class="example-chip">What is Gina Carter's monthly premium?</span>
    <span class="example-chip">Does Gina Carter's policy cover flood damage?</span>
    """, unsafe_allow_html=True)

with col_result:
    if ask_btn and query.strip():
        with st.spinner("Routing and retrieving..."):
            result = support_agent_final(query, customer_name.strip() or None)

        st.markdown(f'<span class="route-badge badge-{result["route"]}">{result["route"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        if result["sources"]:
            sources_list = "<br>".join(f"• {s}" for s in result["sources"])
            st.markdown(f'<div class="sources-box"><b>Sources:</b><br>{sources_list}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#ffffff; border-radius:12px; padding:3rem 2rem;
                    text-align:center; border:1px dashed #d1d5db; margin-top:1rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🛡️</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.4rem;
                        color:#0f1923; margin-bottom:0.5rem;">Ready to help</div>
            <div style="color:#9ca3af; font-size:0.9rem;">
                Ask a policy or account question on the left.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<hr style="border:none; border-top:1px solid #e5e7eb; margin-top:3rem;">
<p style="text-align:center; color:#9ca3af; font-size:0.78rem;">
Built with hybrid (FAISS + BM25) retrieval · Groq (GPT-OSS) · Streamlit &nbsp;|&nbsp;
Policy documents from Axiom Mutual Insurance Company (public wordings) &nbsp;|&nbsp;
Customer data is synthetic &nbsp;|&nbsp; Not affiliated with Axiom Mutual, not insurance advice.
</p>
""", unsafe_allow_html=True)
