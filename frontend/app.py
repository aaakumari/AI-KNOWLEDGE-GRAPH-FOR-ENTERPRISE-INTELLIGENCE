import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pyvis.network import Network
import streamlit.components.v1 as components
import requests

# =========================
# CONFIG
# =========================
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j123"

# =========================
# CONNECT TO NEO4J
# =========================
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]

# =========================
# LOAD DOCUMENTS FROM GRAPH
# =========================
def load_documents_from_graph():
    query = """
    MATCH (n)
    RETURN 
        coalesce(n.name, '') + ' ' +
        reduce(props = '', key IN keys(n) |
            props + key + ': ' + toString(n[key]) + ' ') 
        AS text
    LIMIT 1000
    """
    data = run_query(query)
    texts = [d["text"] for d in data if d["text"] and d["text"].strip() != ""]
    return texts

# =========================
# VECTOR SETUP
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_index():
    texts = load_documents_from_graph()
    if len(texts) == 0:
        return None, []
    model = load_model()
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, texts

model = load_model()
index, documents = build_index()

def semantic_search(query, k=5):
    if index is None or len(documents) == 0:
        return ["⚠ No indexed data found."]
    k = min(k, index.ntotal)
    query_vec = model.encode([query])
    query_vec = np.array(query_vec).astype("float32")
    D, I = index.search(query_vec, k)
    return [documents[i] for i in I[0]]

# =========================
# LLM GENERATION (RAG MODE)
# =========================
def generate_answer(query, context):
    prompt = f"""
You are a strict enterprise assistant.

Answer ONLY using the provided database context.
Do NOT guess.
Do NOT assume.

Database Context:
{context}

User Question:
{query}

Extract and present exact information clearly.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "No response from model.")
    except Exception as e:
        return f"LLM Error: {str(e)}"

# =========================
# GRAPH ANALYTICS (TEXT → CYPHER)
# =========================
def generate_cypher(query):
    prompt = f"""
You are a Neo4j expert.

Database Schema:

Nodes:
- Customer
- Ticket
- Product
- Issue

Relationships:
- (Customer)-[:RAISED]->(Ticket)
- (Ticket)-[:ABOUT]->(Product)
- (Ticket)-[:HAS_ISSUE]->(Issue)
- (Customer)-[:BOUGHT]->(Product)

Convert the user question into a Cypher query.

Rules:
- Use exact labels and relationship names.
- Use aggregation if question asks how many, most, highest, top.
- Only output Cypher query.
- No explanation.

User Question:
{query}
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )
        return response.json()["response"].strip()
    except:
        return None

def run_generated_cypher(cypher_query):
    try:
        return run_query(cypher_query)
    except:
        return []

def format_result(result):
    if not result:
        return "No data found."
    formatted = []
    for r in result:
        if "total_tickets" in r:
            formatted.append(f"{r.get('c.name','Customer')} has raised {r['total_tickets']} tickets.")
        elif "failure_count" in r:
            formatted.append(f"{r.get('p.name','Product')} is the product with the most failures ({r['failure_count']} tickets).")
        elif "issue_count" in r:
            formatted.append(f"{r.get('i.name','Issue')} is the most common issue ({r['issue_count']} tickets).")
        elif "total_products" in r:
            formatted.append(f"{r.get('c.name','Customer')} has bought {r['total_products']} products.")
        else:
            formatted.append(str(r))
    return "\n".join(formatted)

# =========================
# GRAPH VISUALIZATION DATA
# =========================
def fetch_graph_data():
    query = """
    MATCH (n)-[r]->(m)
    RETURN 
        coalesce(toString(n.name), toString(id(n))) AS source,
        type(r) AS relationship,
        coalesce(toString(m.name), toString(id(m))) AS target
    LIMIT 100
    """
    return run_query(query)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("🚀 AI Based Knowledge Graph Builder for Enterprise Intelligence")
st.sidebar.header("System Status")

if index is not None:
    st.sidebar.success("Neo4j Connected")
    st.sidebar.success("Embeddings Loaded")
    st.sidebar.success("Vector Index Ready")
else:
    st.sidebar.error("No Data Found in Neo4j")

# Metrics
col1, col2, col3 = st.columns(3)
total_nodes = len(documents)
vector_size = index.ntotal if index else 0
total_relations = len(fetch_graph_data())
col1.metric("Total Nodes", total_nodes)
col2.metric("Vector Index Size", vector_size)
col3.metric("Graph Relations", total_relations)

# =========================
# QUERY SECTION
# =========================
demo_queries = [
    "Show all tickets raised by Gabrielle Camcho"
    "List the top issues in tickets.",
    "How many products has Angelica Tucker bought?"
]

user_query = st.text_input(
    "Enter your question",
    placeholder="Try queries like:\n" + "\n".join(demo_queries)
)

if st.button("Search") and user_query:
    analytical_keywords = ["how many", "most", "count", "highest", "top"]
    
    if any(word in user_query.lower() for word in analytical_keywords):
        # GRAPH ANALYTICAL MODE
        cypher_query = generate_cypher(user_query)
        if cypher_query:
            results = run_generated_cypher(cypher_query)
            st.write("### 📊 Answer")
            st.success(format_result(results))
        else:
            st.warning("Could not generate Cypher query for this question.")
    else:
        # SEMANTIC SEARCH MODE
        results = semantic_search(user_query)
        context = "\n".join(results)
        answer = generate_answer(user_query, context)
        st.write("### 🤖  Answer")
        st.write(answer)
        st.write("### 📌 Top Relevant Results")
        for r in results:
            st.write("-", r)

    # Always show Graph Visualization
    st.write("### 🌐 Graph Visualization")
    graph_data = fetch_graph_data()
    net = Network(height="500px", width="100%", bgcolor="#111111", font_color="white")
    for row in graph_data:
        source = str(row.get("source","Unknown"))
        target = str(row.get("target","Unknown"))
        relationship = str(row.get("relationship",""))
        net.add_node(source, label=source)
        net.add_node(target, label=target)
        net.add_edge(source, target, label=relationship)
    net.save_graph("graph.html")
    with open("graph.html","r",encoding="utf-8") as f:
        components.html(f.read(), height=550)