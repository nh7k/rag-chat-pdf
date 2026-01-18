import streamlit as st
from PyPDF2 import PdfReader
import io
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Chat with PDFs (Gemini 2)", layout="wide")
MAX_CALLS_PER_MINUTE = 10

# ---------------- SESSION STATE ----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0
if "minute_start" not in st.session_state:
    st.session_state.minute_start = datetime.now()
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- RATE LIMIT ----------------
def can_call_api():
    now = datetime.now()
    if (now - st.session_state.minute_start).seconds >= 60:
        st.session_state.api_calls = 0
        st.session_state.minute_start = now

    if st.session_state.api_calls >= MAX_CALLS_PER_MINUTE:
        st.error("❌ Gemini API quota reached. Wait 60 seconds.")
        return False

    st.session_state.api_calls += 1
    return True

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------- PDF PROCESSING ----------------
def read_pdfs(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(io.BytesIO(pdf.read()))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    embeddings = load_embeddings()
    return FAISS.from_texts(chunks, embeddings)

# ---------------- GEMINI QA ----------------
@st.cache_resource
def get_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.2,
        max_output_tokens=1024,
        google_api_key=api_key
    )

def get_answer(llm, question, context):
    """Generate answer using LLM with context"""
    prompt = f"""You must answer ONLY using the provided context.
If the answer is not present, say: "I cannot find this information in the documents."

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content

# ---------------- UI ----------------
st.title("📚 Chat with PDFs (Gemini 2.0 Flash)")
st.caption("Stable • No 404 • LangChain-Safe")

with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input(
        "Google Gemini API Key",
        type="password"
    )
    st.markdown("[Get API Key](https://aistudio.google.com/app/apikey)")
    st.divider()

    pdfs = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):
        if not api_key:
            st.error("Please enter API key")
        elif not pdfs:
            st.error("Please upload PDFs")
        else:
            with st.spinner("Processing PDFs..."):
                text = read_pdfs(pdfs)
                if not text.strip():
                    st.error("No readable text found in PDFs")
                else:
                    st.session_state.vector_store = build_vector_store(text)
                    st.session_state.processed = True
                    st.success("✅ PDFs processed successfully")

    st.divider()
    st.metric("API Calls (this minute)", st.session_state.api_calls)

# ---------------- CHAT ----------------
if st.session_state.processed:
    question = st.text_input("Ask a question about your PDFs")

    if question:
        if not api_key:
            st.error("API key missing")
        elif not can_call_api():
            pass
        else:
            docs = st.session_state.vector_store.similarity_search(question, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            llm = get_llm(api_key)

            with st.spinner("Thinking..."):
                answer = get_answer(llm, question, context)

            st.session_state.history.append((question, answer))

# ---------------- HISTORY ----------------
if st.session_state.history:
    st.divider()
    st.subheader("Conversation History")

    for q, a in reversed(st.session_state.history):
        st.markdown(f"**❓ {q}**")
        st.markdown(f"{a}")
        st.divider()
