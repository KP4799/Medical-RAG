import sys
import traceback
from pathlib import Path
import streamlit as st
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from services.rag_pipeline import RAGPipeline
from services.upload_service import UploadService

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# Initialize Pipeline
# -------------------------

@st.cache_resource
def load_rag():
    return RAGPipeline()

# -------------------------
# Load Knowledge Base Stats
# -------------------------

def knowledge_base_stats():
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    num_chunks = len(chunks)
    documents = {chunk["source"] for chunk in chunks}
    num_documents = len(documents)
    topics = sorted({chunk["topic"].title() for chunk in chunks})

    knowledge_base = {"num_chunks": num_chunks,
                    "num_documents": num_documents,
                    "topics": topics
                }

    return knowledge_base

pipeline = load_rag()
upload_service = UploadService()
knowledge_base = knowledge_base_stats()

# Uploader key to make a new uploader widget to remove the previous selection
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your Medical AI Assistant.\n\nAsk me anything about the uploaded medical literature. I'll answer using only the information available in the knowledge base and cite the relevant sources."
            )
        }
    ]

# -------------------------
# Helper Functions
# -------------------------

def display_chunks(chunks):
    """Display retrieved evidence."""
    if not chunks:
        return

    with st.expander("🔍 Retrieved Evidence"):
        for chunk in chunks:
            st.markdown(
                f"""
**Source:** {chunk['source']}

**Page:** {chunk['page']}

**Topic:** {chunk['topic']}
"""
            )

            st.write(chunk["text"])
            st.divider()


def display_rewritten_question(question):
    if not question:
        return
    with st.expander("Interpreted Question"):
        st.write(question)

# -------------------------
# Sidebar
# -------------------------

with st.sidebar:
    st.title("📄 Document Management")

    with st.sidebar.expander("📤 Upload Document", expanded=False):
        uploaded_file = st.file_uploader("Upload Medical PDF", type=["pdf"], key=f"uploader_{st.session_state.uploader_key}")

        topic_options = knowledge_base['topics'] + ["➕ Create New Topic"]
        selected_topic = st.selectbox("Selected Topic", topic_options)
        new_topic = ''

        if selected_topic == "➕ Create New Topic":
            new_topic = st.text_input("New Topic Name")

        upload_clicked = st.button("Upload Document",use_container_width=True)

        if upload_clicked:
            if uploaded_file is None:
                st.error("Please select a PDF.")

            elif selected_topic == "➕ Create New Topic" and not new_topic.strip():
                st.error("Please enter a topic name.")
                
            else:
                if selected_topic == "➕ Create New Topic":
                    topic = new_topic.strip()
                else:
                    topic = selected_topic
                try:
                    with st.status("Processing document", expanded=True) as status:

                        st.write("📄 Saving PDF and updating knowledge base")
                        stats = upload_service.upload_pdf(uploaded_file, topic)

                        status.update(
                            label="✅ Document indexed successfully!",
                            state="complete"
                        )
                    st.session_state["upload_success"] = stats
                    st.session_state.uploader_key += 1

                    load_rag.clear()
                    st.rerun()

                except FileExistsError as e:
                    st.warning(str(e))

                except Exception as e:
                    # st.error(str(e))
                    st.exception(traceback.format_exc())

    # To make the success message persist the rerun and show it only once
    if "upload_success" in st.session_state:
        stats = st.session_state.pop("upload_success")

        st.success(f"""
    ✅ Document indexed successfully!

    📄 Pages indexed: {stats['pages']}

    🧩 Chunks created: {stats['chunks']}
    """
        )

    st.divider()
    st.title("📚 Knowledge Base")

    st.metric("Documents", knowledge_base['num_documents'])
    st.metric("Chunks", knowledge_base['num_chunks'])

    st.divider()

    st.markdown("### 🏥 Topics")

    for topic in knowledge_base['topics']:
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
    "This system is for educational purposes only and does not provide medical advice."
)

# -------------------------
# Display Chat History
# -------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_rewritten_question(message.get("rewritten_question", ""))
            display_chunks(message.get("chunks", []))

# -------------------------
# Chat Input
# -------------------------

if prompt := st.chat_input("Ask a medical question"):

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Understanding your question and searching the knowledge base"):
            try:
                result = pipeline.answer_question(prompt,chat_history=st.session_state.messages)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        answer = result.get("answer", "No answer generated.")
        llm_available = result.get("llm_available", True)
        sources = result.get("sources", [])
        chunks = result.get("chunks", [])
        rewritten_question = result.get("rewritten_question","")

        # Display answer
        st.markdown(answer)

        if rewritten_question:
            display_rewritten_question(rewritten_question)

        if not llm_available:
            st.warning(
                "⚠️ I couldn't generate a summarized answer because the language model is temporarily unavailable."
                "However, I successfully searched the medical knowledge base and found the most relevant passages below."
                "You can review these retrieved excerpts directly."
                )
            
        display_chunks(chunks)

    # Save assistant response to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "rewritten_question": rewritten_question,
            "sources": sources,
            "chunks": chunks
        }
    )

st.markdown("---")
st.caption(
    "Educational use only. Always consult qualified healthcare professionals for medical advice."
)

#python -m streamlit run app/streamlit_app.py --server.fileWatcherType none 
