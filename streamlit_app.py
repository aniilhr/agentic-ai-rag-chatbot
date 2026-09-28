"""Streamlit chat UI for the Agentic AI RAG chatbot.

    streamlit run streamlit_app.py

Runs the LangGraph workflow in-process; the side panel shows the retrieved
chunks and scores for the most recent answer.
"""

from __future__ import annotations

import streamlit as st

from src.graph import build_rag_graph

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="Connecting to Pinecone and OpenAI…")
def load_graph():
    return build_rag_graph()


st.title("🤖 Agentic AI eBook Chatbot")
st.caption("Answers are generated strictly from the Agentic AI eBook. Off-topic questions are declined.")

try:
    graph = load_graph()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.last_result = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about Agentic AI…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            result = graph.invoke({"question": prompt})
        st.markdown(result["answer"])
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.last_result = result

with st.sidebar:
    st.header("Retrieval details")
    result = st.session_state.get("last_result")
    if not result:
        st.info("Ask a question to see the retrieved context.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Confidence", f"{result.get('score', 0.0):.2f}")
        col2.metric("Retrieval", f"{result.get('retrieval_score', 0.0):.2f}")
        st.write("Grounded answer" if result.get("grounded") else "Declined — not in document")

        cited = set(result.get("cited_chunks", []))
        for chunk in result.get("context", []):
            label = f"[{chunk['rank']}] page {chunk['page']} · sim {chunk['similarity']:.3f}"
            if chunk["rank"] in cited:
                label += " · cited"
            with st.expander(label, expanded=chunk["rank"] in cited):
                st.write(chunk["text"])

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()
