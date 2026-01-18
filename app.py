import streamlit as st
from PyPDF2 import PdfReader
import io
from datetime import datetime
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Chat with PDFs", layout="wide")
MAX_CALLS_PER_MINUTE = 10  # Reduced from 15

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
if "llm" not in st.session_state:
    st.session_state.llm = None
if "last_call_time" not in st.session_state:
    st.session_state.last_call_time = None

# ---------------- RATE LIMIT ----------------
def can_call_api():
    now = datetime.now()
    elapsed = (now - st.session_state.minute_start).total_seconds()
    
    # Reset counter after 1 minute
    if elapsed >= 60:
        st.session_state.api_calls = 0
        st.session_state.minute_start = now
        elapsed = 0

    # Check rate limit
    if st.session_state.api_calls >= MAX_CALLS_PER_MINUTE:
        remaining = 60 - int(elapsed)
        st.error(f"❌ Rate limit reached. Wait {remaining} seconds.")
        return False

    # Add 2-second delay between calls to avoid quota issues
    if st.session_state.last_call_time:
        time_since_last = (now - st.session_state.last_call_time).total_seconds()
        if time_since_last < 2:
            time.sleep(2 - time_since_last)

    return True

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# ---------------- PDF PROCESSING ----------------
def read_pdfs(pdfs):
    text = ""
    for pdf in pdfs:
        try:
            pdf_bytes = pdf.read()
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
    return text

def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced from 1000
        chunk_overlap=100,  # Reduced from 150
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    if not chunks:
        raise ValueError("No text chunks created from PDFs")
    
    embeddings = load_embeddings()
    return FAISS.from_texts(chunks, embeddings)

# ---------------- GEMINI LLM ----------------
def initialize_llm(api_key):
    """Initialize LLM with error handling"""
    try:
        if st.session_state.llm is None or st.session_state.get('current_api_key') != api_key:
            st.session_state.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # Stable model
                temperature=0.2,  # More focused answers
                max_output_tokens=256,  # Reduced from 512
                google_api_key=api_key,
                request_options={"timeout": 30}
            )
            st.session_state.current_api_key = api_key
        return st.session_state.llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

def get_answer(llm, question, context):
    """Generate answer with error handling and retry logic"""
    if llm is None:
        return "Error: LLM not initialized"
    
    # Truncate context to save tokens
    max_context_length = 2000  # Reduced from 3000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    prompt = f"""Answer briefly using ONLY this context.
If not found, say: "Not in documents."

Context: {context}

Q: {question}
A:"""
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle quota exceeded
            if "quota" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"⏳ Quota limit hit. Waiting 5 seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)
                    continue
                else:
                    return "❌ API quota exceeded. Please wait a minute and try again, or check your API key billing."
            
            # Handle other rate limits
            elif "rate" in error_msg:
                return "❌ Rate limit reached. Please wait 60 seconds before asking again."
            
            # Handle other errors
            else:
                return f"❌ Error: {str(e)[:100]}"
    
    return "❌ Failed after multiple retries. Please try again later."

# ---------------- UI ----------------
st.title("📚 Chat with PDFs")
st.caption("💡 Optimized for Gemini Free Tier")

with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        help="Get your key from Google AI Studio"
    )
    
    if api_key:
        st.success("✅ API Key provided")
    else:
        st.info("👆 Enter your API key to start")
    
    st.markdown("[🔗 Get API Key](https://aistudio.google.com/app/apikey)")
    st.divider()

    pdfs = st.file_uploader(
        "📄 Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files"
    )

    if st.button("🚀 Process PDFs", type="primary", disabled=not api_key):
        if not api_key:
            st.error("⚠️ Please enter your API key first")
        elif not pdfs:
            st.error("⚠️ Please upload at least one PDF")
        else:
            with st.spinner("📖 Processing PDFs..."):
                try:
                    text = read_pdfs(pdfs)
                    
                    if not text.strip():
                        st.error("❌ No readable text found in PDFs")
                    elif len(text) < 100:
                        st.warning("⚠️ Very little text extracted. PDFs may be image-based.")
                    else:
                        st.session_state.vector_store = build_vector_store(text)
                        st.session_state.processed = True
                        
                        # Initialize LLM after processing
                        llm = initialize_llm(api_key)
                        if llm:
                            char_count = len(text)
                            st.success(f"✅ Successfully processed {len(pdfs)} PDF(s)")
                            st.info(f"📊 Extracted {char_count:,} characters")
                        else:
                            st.error("Failed to initialize AI model")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.processed = False

    st.divider()
    
    # API Usage Display
    st.subheader("📊 API Usage")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Calls Used", st.session_state.api_calls)
    with col2:
        st.metric("Limit/Min", MAX_CALLS_PER_MINUTE)
    
    # Progress bar
    progress = min(st.session_state.api_calls / MAX_CALLS_PER_MINUTE, 1.0)
    st.progress(progress)
    
    if st.button("🔄 Reset Counter"):
        st.session_state.api_calls = 0
        st.session_state.minute_start = datetime.now()
        st.session_state.last_call_time = None
        st.rerun()

# ---------------- CHAT INTERFACE ----------------
if st.session_state.processed:
    st.divider()
    
    # Chat input
    question = st.text_input(
        "💬 Ask a question about your PDFs",
        placeholder="What is this document about?",
        key="question_input",
        disabled=not api_key
    )

    if question and question.strip():
        if not api_key:
            st.error("⚠️ API key missing")
        elif not can_call_api():
            pass  # Error already shown in can_call_api()
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Retrieve relevant documents
                    docs = st.session_state.vector_store.similarity_search(
                        question, 
                        k=2  # Reduced from 3 to minimize tokens
                    )
                    
                    # Combine document contents
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Get or initialize LLM
                    llm = initialize_llm(api_key)
                    
                    if llm:
                        # Get answer
                        answer = get_answer(llm, question, context)
                        
                        # Update tracking
                        st.session_state.api_calls += 1
                        st.session_state.last_call_time = datetime.now()
                        
                        # Save to history
                        st.session_state.history.append((question, answer))
                        
                        # Clear input
                        st.rerun()
                    else:
                        st.error("LLM not available")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Please upload and process PDFs using the sidebar to get started")

# ---------------- CONVERSATION HISTORY ----------------
if st.session_state.history:
    st.divider()
    st.subheader("💬 Conversation History")
    
    # Add clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
    
    # Display history in reverse order (newest first)
    for idx, (q, a) in enumerate(reversed(st.session_state.history)):
        with st.container():
            st.markdown(f"**❓ Question {len(st.session_state.history) - idx}:**")
            st.info(q)
            st.markdown("**💡 Answer:**")
            st.success(a)
            st.divider()
