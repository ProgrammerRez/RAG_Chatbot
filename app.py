import os
import tempfile
import streamlit as st

# LangChain / LangGraph
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# OCR libraries
from pdf2image import convert_from_path
import pytesseract


# =====================================================
# State Definition
# =====================================================
class State(TypedDict):
    """Defines the structure of a RAG conversation state."""
    question: str
    context: List[Document]
    answer: str


# =====================================================
# Streamlit Page Setup
# =====================================================
st.set_page_config(
    page_title='📄 RAG QNA with Document Support',
    layout='wide',
    initial_sidebar_state='expanded'
)
st.title('📄 RAG QNA with Document Support')
st.caption("Upload PDF files and ask questions. Powered by Groq + LangChain.")


# =====================================================
# OCR Helper
# =====================================================
def ocr_pdf_to_docs(file_path: str) -> List[Document]:
    """
    Convert scanned PDFs to text using Tesseract OCR.
    
    Args:
        file_path (str): Path to PDF file.

    Returns:
        List[Document]: Extracted text as Document objects.
    """
    pages = convert_from_path(file_path)
    docs = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="eng")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs


# =====================================================
# Text Splitting Helper
# =====================================================
def text_splits(uploaded_files) -> List[Document]:
    """
    Extract and split text from uploaded PDFs.
    Uses OCR fallback if PDF is scanned.

    Args:
        uploaded_files: List of uploaded PDF files.

    Returns:
        List[Document]: List of text chunks as Documents.
    """
    splits = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Fallback to OCR for scanned PDFs
            if all("studyplusplus" in d.page_content.lower() for d in docs):
                st.warning(f"⚠️ Falling back to OCR for **{uploaded_file.name}** (scanned PDF detected).")
                docs = ocr_pdf_to_docs(tmp_path)
        finally:
            os.remove(tmp_path)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits.extend(splitter.split_documents(docs))
    return splits


# =====================================================
# Embeddings + Vector Store
# =====================================================
def get_embeddings():
    """Load HuggingFace embeddings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def get_vectorstore(embeddings):
    """Initialize an in-memory Chroma vector store."""
    return Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=None
    )


# =====================================================
# RAG Graph Functions
# =====================================================
def retrieve(state: State, vector_store):
    """Retrieve relevant documents from vector store."""
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}

def generate(state: State):
    """Generate answer from LLMChain using question + context."""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    response = st.session_state.llm_chain.invoke({
        "question": state["question"],
        "context": docs_content
    })
    return {"answer": response["text"]}


# =====================================================
# Sidebar: API Key + Session Handling
# =====================================================
if "memories" not in st.session_state:
    st.session_state.memories = {}

with st.sidebar:
    st.title("⚙️ Configuration")
    st.header("🤖 LLM Settings")
    st.info("Don’t have an API key? Get yours from: https://console.groq.com/keys")
    groq_key = st.text_input(label='🔑 Groq API Key')
    model_name = st.selectbox(
        "Choose Model:",
        ["qwen/qwen3-32b", "gemma2-9b-it", "openai/gpt-oss-120b"]
    )
    temp = st.slider('Temperature',min_value=0.0,max_value=2.0,value=0.7,step=0.1)
    st.header("🗂 Session Management")
    if st.button("➕ New Session"):
        new_id = f"session_{len(st.session_state.memories) + 1}"
        st.session_state.memories[new_id] = ConversationBufferMemory(
            memory_key="chat_history", input_key='question', return_messages=True
        )
        st.session_state.session_id = new_id

    existing_sessions = list(st.session_state.memories.keys())
    if not existing_sessions:
        st.session_state.memories["default"] = ConversationBufferMemory(
            memory_key="chat_history", input_key='question', return_messages=True
        )
        st.session_state.session_id = "default"

    session_id = st.selectbox(
        "Select Session:",
        options=list(st.session_state.memories.keys()),
        index=len(st.session_state.memories) - 1
    )
    st.session_state.session_id = session_id

if not groq_key:
    st.warning("⚠️ Please enter your Groq API key to continue.")
    st.stop()


# =====================================================
# File Upload + Processing
# =====================================================
uploaded_files = st.file_uploader("📂 Upload your PDF files:", accept_multiple_files=True)

if st.button("⚡ Process Files") and uploaded_files:
    with st.spinner("⏳ Processing files, please wait..."):
        # Setup chain
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatGroq(
            model=model_name,
            api_key=groq_key,
            temperature=temp,
        )
        parser = StrOutputParser()
        chain = llm | parser
        st.session_state.llm_chain = LLMChain(
            llm=chain,
            prompt=prompt,
            memory=st.session_state.memories[session_id]
        )

        # Vector store + splits
        embeddings = get_embeddings()
        vectorstore = get_vectorstore(embeddings)
        splits = text_splits(uploaded_files)
        vectorstore.add_documents(splits)

        # Build RAG graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", lambda state: retrieve(state, vectorstore))
        graph_builder.add_node("generate", generate)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        st.session_state.graph = graph_builder.compile()

        st.success("✅ Documents processed successfully! You can now ask questions.")


# =====================================================
# Show Conversation History
# =====================================================
if history := st.session_state.memories[st.session_state.session_id].chat_memory.messages:
    for msg in history:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif msg.type == "ai":
            with st.chat_message("assistant"):
                st.markdown(msg.content)


# =====================================================
# Chat UI
# =====================================================
if "graph" in st.session_state:
    if user_question := st.chat_input("💬 Ask a question about your documents:"):
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching for answer..."):
                result = st.session_state.graph.invoke({"question": user_question})

                with st.expander("📖 Retrieved Context (Preview)"):
                    for doc in result["context"]:
                        st.write(f"{doc.page_content[:500]}...")

                st.subheader("💡 Answer:")
                st.markdown(result["answer"])
