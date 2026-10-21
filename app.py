"""
app.py

Main Streamlit Application for UniStudy AI.
Provides a local study assistant interface with observable agent execution traces.
"""

import streamlit as st
from typing import List, Dict, Any

from llm.ollama_client import OllamaClient
from tools.document_reader import read_document
from utils.text_processing import chunk_document
from agents.study_agent import StudyAgent

# Page configuration
st.set_page_config(
    page_title="UniStudy AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}  # filename -> metadata dict

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []  # Flat list of all chunk dicts


# Cached client and agent
@st.cache_resource
def get_ollama_client(model_name: str = "llama3.2:3b") -> OllamaClient:
    return OllamaClient(model=model_name)


client = get_ollama_client()
agent = StudyAgent(client=client)

# ---------------------------------------------------------
# Sidebar: System Health & Document Upload
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎓 UniStudy AI")
    st.caption("Local Academic Document & Study Assistant")

    st.subheader("1. System Health")
    health = client.check_health()
    if health["is_running"] and health["model_available"]:
        st.success(f"● Ollama Active ({client.model})", icon="✅")
    elif health["is_running"] and not health["model_available"]:
        st.warning(f"Ollama running, but model '{client.model}' not found.\nRun `ollama pull {client.model}`.")
    else:
        st.error("Ollama is offline. Start the Ollama desktop service.")

    st.divider()

    st.subheader("2. Upload Study Notes")
    uploaded_files = st.file_uploader(
        "Upload lecture PDFs, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file_obj in uploaded_files:
            fname = file_obj.name
            if fname not in st.session_state.uploaded_docs:
                with st.spinner(f"Ingesting {fname}..."):
                    doc_data = read_document(file_obj, fname)
                    if doc_data.get("error"):
                        st.error(f"Error reading {fname}: {doc_data['error']}")
                    else:
                        # Chunk the document
                        chunks = chunk_document(doc_data, chunk_size_words=300, chunk_overlap_words=50)
                        st.session_state.uploaded_docs[fname] = {
                            "total_pages": doc_data["total_pages"],
                            "chunk_count": len(chunks),
                            "file_type": doc_data["file_type"]
                        }
                        st.session_state.all_chunks.extend(chunks)

        st.success(f"{len(st.session_state.uploaded_docs)} document(s) indexed.")

    # Show active documents
    if st.session_state.uploaded_docs:
        st.markdown("**Indexed Documents:**")
        for doc_name, info in st.session_state.uploaded_docs.items():
            st.markdown(f"- 📄 `{doc_name}` ({info['total_pages']} pg, {info['chunk_count']} chunks)")

    st.divider()

    # Session Management
    st.subheader("3. Session Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_docs = {}
            st.session_state.all_chunks = []
            st.rerun()

# ---------------------------------------------------------
# Main Chat & Task Interface
# ---------------------------------------------------------
st.header("Academic Assistant Workspace")
st.markdown("Ask questions about your uploaded documents, generate revision quizzes, or request structured summaries.")

# Quick task prompt chips
chip_col1, chip_col2, chip_col3 = st.columns(3)
with chip_col1:
    if st.button("📝 Summarize Uploaded Notes", use_container_width=True):
        st.session_state.pending_prompt = "Please summarize the core concepts from my uploaded notes."
with chip_col2:
    if st.button("❓ Generate Practice Quiz", use_container_width=True):
        st.session_state.pending_prompt = "Generate a 3-question multiple choice practice quiz from my notes."
with chip_col3:
    if st.button("✍️ Evaluate My Understanding", use_container_width=True):
        st.session_state.pending_prompt = "Evaluate my answer: Gradient descent is when a model takes steps to find minimum error."

# Render existing chat conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # If this assistant turn contains a tool execution trace, render it inside an expander
        if msg.get("trace"):
            with st.expander("🔍 View Agent Execution Trace", expanded=False):
                for item in msg["trace"]:
                    status_badge = "🟢 SUCCESS" if item["status"] == "success" else "🔴 FAILED"
                    st.markdown(
                        f"**Step {item['step']} | Tool:** `{item['tool']}` &nbsp;&nbsp; `{status_badge}`\n\n"
                        f"- **Input:** {item['input_summary']}\n"
                        f"- **Output:** {item['result_summary']}\n"
                    )

        # If citations/sources were pulled, display them
        if msg.get("sources"):
            with st.expander("📚 Cited Source Passages", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**From `{src.get('filename')}` — Page {src.get('page_num')}** (Score: {src.get('score', 'N/A')})")
                    st.info(src.get("text", ""))

# ---------------------------------------------------------
# Input Handling
# ---------------------------------------------------------
user_input = st.chat_input("Ask a question about your documents, request a summary, or type a study query...")

# Check if a chip button triggered a pending prompt
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_input:
    # 1. Render User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Execute Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent analyzing request and selecting tools..."):
            response = agent.run(
                user_query=user_input,
                document_chunks=st.session_state.all_chunks
            )

        st.markdown(response["answer"])

        # Render Agent Trace
        if response.get("trace"):
            with st.expander("🔍 View Agent Execution Trace", expanded=True):
                for item in response["trace"]:
                    status_badge = "🟢 SUCCESS" if item["status"] == "success" else "🔴 FAILED"
                    st.markdown(
                        f"**Step {item['step']} | Tool:** `{item['tool']}` &nbsp;&nbsp; `{status_badge}`\n\n"
                        f"- **Input:** {item['input_summary']}\n"
                        f"- **Output:** {item['result_summary']}\n"
                    )

        # Render Sources
        if response.get("sources"):
            with st.expander("📚 Cited Source Passages", expanded=False):
                for src in response["sources"]:
                    st.markdown(
                        f"**From `{src.get('filename')}` — Page {src.get('page_num')}** (Score: {src.get('score', 'N/A')})")
                    st.info(src.get("text", ""))

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "trace": response.get("trace", []),
        "sources": response.get("sources", [])
    })