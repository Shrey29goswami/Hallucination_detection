from __future__ import annotations

import json
from pathlib import Path
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Hallucination Detection & Alignment System", layout="wide")
st.title("Hallucination Detection & Alignment System")
st.caption("RAG + HNSW + GraphRAG + guardrails + evaluation")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    use_graph = st.checkbox("Use GraphRAG expansion", value=True)
    top_k = st.slider("Top K evidence chunks", min_value=3, max_value=10, value=6)

st.subheader("1) Upload documents")
uploaded = st.file_uploader("Upload one or more .txt files", type=["txt"], accept_multiple_files=True)
if st.button("Ingest documents") and uploaded:
    files = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded]
    response = requests.post(f"{api_url}/ingest", files=files, timeout=120)
    st.json(response.json())

st.subheader("2) Ask a question")
question = st.text_area("Question", value="What controls reduce hallucination in enterprise RAG systems?")
if st.button("Run pipeline"):
    payload = {"question": question, "use_graph": use_graph, "top_k": top_k}
    response = requests.post(f"{api_url}/ask", json=payload, timeout=180)
    data = response.json()
    st.markdown("### Answer")
    st.write(data["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval confidence", data["scores"]["retrieval_confidence"])
    col2.metric("Groundedness", data["scores"]["groundedness"])
    col3.metric("Risk-adjusted trust", data["scores"]["risk_adjusted_trust"])

    st.markdown("### Validation")
    st.json(data["validation"])
    st.markdown("### Guardrails")
    st.json(data["guardrails"])
    st.markdown("### Evidence")
    for item in data["citations"]:
        with st.expander(f"{item['title']} | {item['chunk_id']} | {item['retrieval_type']} | score={item['score']:.3f}"):
            st.write(item["text"])

st.subheader("3) Run evaluation")
default_eval = Path("data/eval_dataset.json")
if st.button("Run benchmark"):
    response = requests.post(f"{api_url}/evaluate", json={"dataset_path": str(default_eval)}, timeout=300)
    st.json(response.json())
