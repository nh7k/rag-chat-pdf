# 📚 PDF RAG Chat Application (Streamlit + Gemini)

## 🚀 Overview

This project is a **Retrieval-Augmented Generation (RAG)** based web application that allows users to **upload PDF documents and chat with them** using natural language.

The application uses:

* **Streamlit** for the web interface
* **FAISS** for vector similarity search
* **HuggingFace sentence-transformers** for embeddings
* **Google Gemini (Generative AI)** for intelligent question answering

It is designed to be **hackathon-ready, scalable, and production-aligned**.

---

## 🎯 Why This Project Is Useful

### 🔍 Problem It Solves

Large documents such as:

* Research papers
* Government reports
* Legal documents
* Academic notes

are difficult and time-consuming to read fully. Users often want **specific answers**, not entire documents.

### ✅ Solution Provided

This app enables users to:

* Upload multiple PDFs
* Ask natural language questions
* Receive **accurate answers strictly from the document content**

No hallucination. No guessing.

---

## 🧠 How It Works (Architecture)

1. **PDF Upload**

   * PDFs are uploaded via Streamlit UI

2. **Text Extraction**

   * Text is extracted using `PyPDF2`

3. **Chunking**

   * Text is split into overlapping chunks using `RecursiveCharacterTextSplitter`

4. **Vector Embeddings**

   * Chunks are converted into embeddings using:

     * `sentence-transformers/all-MiniLM-L6-v2`

5. **Vector Store (FAISS)**

   * Embeddings are stored and searched efficiently

6. **Question Answering (RAG)**

   * Relevant chunks are retrieved
   * Gemini model answers **only from retrieved context**

---

## 🛠 Tech Stack

| Component   | Technology                        |
| ----------- | --------------------------------- |
| Frontend    | Streamlit                         |
| LLM         | Google Gemini API                 |
| Embeddings  | HuggingFace Sentence Transformers |
| Vector DB   | FAISS                             |
| PDF Parsing | PyPDF2                            |
| Language    | Python 3.10+                      |

---

## 📦 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/rag-chat-pdf.git
cd rag-chat-pdf
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate  # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Setup

1. Go to: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Generate a **Google Gemini API key**
3. Enter the key inside the app sidebar (not hardcoded)

⚠️ API keys are **never stored or pushed to GitHub**

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 🧪 Example Use Cases

### 📄 Students

* Ask questions from lecture notes
* Summarize academic PDFs

### 🧑‍⚖️ Legal & Policy

* Query legal documents
* Extract clauses or rules

### 🏢 Enterprises

* Internal document Q&A
* Knowledge-base assistant

### 🏆 Hackathons

* Demonstrates:

  * RAG
  * LLM integration
  * Vector databases

---

## 🔐 Security & Reliability

* No API keys in code
* Rate limiting implemented
* Answers restricted to document context
* Local embeddings (cost-efficient)

---

## 🚧 Current Limitations

* Free Gemini API quota limits
* PDF text extraction depends on document quality
* No OCR for scanned PDFs (future scope)

---

## 🔮 Future Enhancements

* OCR support for scanned PDFs
* User authentication
* Chat history persistence
* Multi-model support (OpenAI, Claude)
* Deployment on Streamlit Cloud / AWS

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 👤 Author

**Nitish Kumar**
B.Tech (IT) | AI/ML Enthusiast


---

## ⭐ If you find this project useful

Give it a ⭐ on GitHub and feel free to contribute!
