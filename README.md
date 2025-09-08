# 📄 RAG QnA with Document Support

[🚀 Live Demo](https://rag-chatbot-session-based-pdf-support.streamlit.app)  

A **professional, session-based RAG (Retrieval-Augmented Generation) chatbot** that allows users to upload PDF documents and ask questions. Powered by **Groq + LangChain**, this tool supports both standard and scanned PDFs (via OCR) and provides context-aware answers in real time.  

---

## ✨ Features

- **📂 PDF Upload & Processing**  
  Upload multiple PDFs for analysis. Scanned PDFs are automatically converted to text using OCR.

- **💬 Session-Based Conversations**  
  Maintain separate chat sessions with persistent memory for a seamless question-answer experience.

- **🧠 Contextual Q&A with RAG**  
  Uses FAISS vector store and embeddings to retrieve the most relevant document snippets before generating answers.

- **🤖 Groq-Powered LLM Integration**  
  Supports multiple models, including `qwen/qwen3-32b`, `gemma2-9b-it`, and `openai/gpt-oss-120b`.

- **🖥 Interactive Chat Interface**  
  Ask questions about your documents and get precise, contextual answers along with retrieved content previews.

- **⚙️ Customizable LLM Settings**  
  Control model selection, temperature, and session management directly from the sidebar.

---

## 🛠 How It Works

1. **📤 Upload PDFs**  
   The app accepts both standard and scanned PDFs. For scanned PDFs, **Tesseract OCR** extracts text.

2. **✂️ Text Splitting & Embeddings**  
   Documents are split into smaller chunks for effective retrieval. Embeddings are generated using **HuggingFace sentence-transformers**.

3. **📚 Vector Store Initialization**  
   Chunks are stored in an **FAISS vector store** for similarity-based retrieval.

4. **🔄 RAG Graph Pipeline**  
   - **Retrieve:** Find the most relevant document chunks.  
   - **Generate:** Use the LLM to answer questions based on retrieved context.

5. **💬 Chat Interface**  
   Users can interact in a session-based chat interface and see both answers and the relevant document snippets.

---

## ⚡ Installation

```bash
git clone <repository-url>
cd rag-chatbot-session-based-pdf-support
pip install -r requirements.txt
streamlit run app.py
````

**Requirements:**

* 🐍 Python ≥ 3.10
* 🖥 Streamlit
* 📦 LangChain & LangGraph
* 🔑 Groq API Key
* 📄 PyPDF2, pdf2image, pytesseract
* ⚡ FAISS & HuggingFace Transformers

---

## ⚙️ Configuration

1. **🔑 Groq API Key:**
   Enter your API key in the sidebar. Get one [here](https://console.groq.com/keys).

2. **🧩 Select Model:**
   Choose the desired LLM for generating answers.

3. **🌡 Adjust Temperature:**
   Control randomness in responses (0.0 = deterministic, 2.0 = creative).

4. **🗂 Session Management:**

   * ➕ Create a new session or select an existing one.
   * 💾 Each session retains chat history and memory for context-aware responses.

---

## 🚀 Usage

1. 📂 Upload one or more PDF files.
2. ⚡ Click **Process Files** to index documents.
3. 💬 Ask questions in the chat input.
4. 📖 View retrieved context and AI-generated answers.
5. 🔄 Switch or create sessions for different topics or users.

---

## 🛠 Technologies Used

* **LangChain & LangGraph:** RAG workflow orchestration
* **Groq LLM:** High-performance language model inference
* **FAISS:** Efficient similarity search for document chunks
* **HuggingFace Embeddings:** Convert text to dense vectors
* **Tesseract OCR:** Convert scanned PDFs to text
* **Streamlit:** Frontend interface for interactive chat

---

## 🌐 Demo

Access the live demo here: [RAG Chatbot with Session-Based PDF Support](https://rag-chatbot-session-based-pdf-support.streamlit.app)

