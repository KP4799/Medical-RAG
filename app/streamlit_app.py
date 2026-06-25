import streamlit as st
import json
from services.rag_pipeline import RAGPipeline

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# Load Knowledge Base Stats
# -------------------------

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

num_chunks = len(chunks)
documents = {chunk["source"] for chunk in chunks}
num_documents = len(documents)
topics = sorted({chunk["topic"].title() for chunk in chunks})

# -------------------------
# Initialize RAG
# -------------------------

@st.cache_resource
def load_rag():
    return RAGPipeline()

rag = load_rag()

# -------------------------
# Sidebar
# -------------------------

with st.sidebar:
    st.title("📚 Knowledge Base")

    st.metric("Documents", num_documents)
    st.metric("Chunks", num_chunks)

    st.divider()

    st.markdown("### 🏥 Topics")

    for topic in topics:
        st.markdown(f"- {topic}")

    st.divider()

    st.markdown("### ⚙️ System")

    st.markdown("""
    **Embedding Model:**
    all-MiniLM-L6-v2
    """)

    st.markdown("""
    **Vector Database:**
    FAISS
    """)

    st.markdown("""
    **LLM:**
    Gemini 2.5 Flash
    """)

    st.divider()
    st.info(
        "This assistant answers questions only "
        "from the uploaded medical documents."
    )

# -------------------------
# Main UI
# -------------------------

st.title("🩺 Medical AI Assistant")

st.warning(
    "This system is for educational purposes only and "
    "does not provide medical advice."
)

question = st.text_area(
    "Ask a medical question:",
    height=100
)

if st.button("Ask"):
    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    with st.spinner("Searching knowledge base..."):
        result = rag.answer_question(question)

    # -------------------------
    # Answer
    # -------------------------

    st.subheader("Answer")
    st.write(result["answer"])

    # -------------------------
    # Sources
    # -------------------------

    if result["sources"]:
        st.subheader("Sources")

        for source in result["sources"]:
            with st.container():
                st.info(
                    f"""
                    📄 {source['source']}

                    Page: {source['page']}
                    | Topic: {source['topic']}
                    """
                )

    # -------------------------
    # Evidence
    # -------------------------

    if result["chunks"]:
        with st.expander("Show Retrieved Evidence"):
            for i, chunk in enumerate(result["chunks"],start=1):
                st.markdown(
                    f"### Evidence {i}"
                )

                st.markdown(
                    f"""
                    **Source:** {chunk['source']}
                    **Page:** {chunk['page']}
                    **Topic:** {chunk['topic']}
                    """
                )

                st.write(chunk["text"])
                st.divider()

# -------------------------
# Footer
# -------------------------

st.markdown("---")
st.caption(
    "Educational use only. "
    "Always consult qualified healthcare professionals"
    "for medical advice."
)