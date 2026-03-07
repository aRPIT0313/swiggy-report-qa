
# 🍜 Swiggy Annual Report — RAG Q&A System

A Retrieval-Augmented Generation (RAG) project that lets you ask natural language questions about the **Swiggy Annual Report** and get answers directly from the document.

This project is fully open-source, uses **TF-IDF + LSA** for retrieval, and does **not require PyTorch or GPU**.


## 📄 Document Source

| Field               | Details                                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Document**        | Swiggy Annual Report (FY2024)                                                                                  |
| **Format**          | PDF                                                                                                            |
| **Source URL**      | [Swiggy Investor Relations](https://ir.swiggy.com/annual-reports)                                              |
| **Direct Download** | [Swiggy Annual Report PDF](https://ir.swiggy.com/sites/default/files/2025-05/Swiggy-Annual-Report-2024-25.pdf) |


## 🏗 Architecture

PDF
 │
 ▼
[Document Processor]   ← pypdf extraction + text cleaning + chunking
 │ chunks.json
 ▼
[Vector Store]         ← TF-IDF vectorization (scikit-learn)
 │ vector_store.pkl     + Cosine similarity search
 ▼
[RAG Engine]
 │
 ├─ retrieve top-K chunks (default k=5)
 │
 └─ [Groq API] ──► generated answers from retrieved chunks
      OR
      [Retrieval-only] ──► show top chunks if no API key


### Components

| File                    | Purpose                                                  |
| ----------------------- | -------------------------------------------------------- |
| `document_processor.py` | PDF → pages → clean text → overlapping chunks            |
| `vector_store.py`       | TF-IDF index + cosine similarity + keyword reranking     |
| `rag_engine.py`         | Retrieve chunks + send to Groq API for answer generation |
| `setup.py`              | One-time PDF processing + index building                 |
| `app.py`                | Web UI using Flask                                       |
| `requirements.txt`      | Python dependencies                                      |
| `data/`                 | Auto-generated folder for chunks and vector store        |


## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies used:

* `pypdf` → PDF text extraction
* `numpy` → Numerical arrays
* `scipy` → Sparse matrices & vector operations
* `scikit-learn` → TF-IDF + LSA + cosine similarity
* `Flask` → Web interface
* `requests` → Sending queries to Groq API

No PyTorch, GPU, or paid embedding APIs are needed.


### 2. Process the PDF

```bash
python build_index.py --pdf "path/Swiggy_Annual_Report_2023.pdf"
```

This will:

* Extract text from all PDF pages
* Clean text and split into overlapping chunks (default 600 words/chunk, 100 overlap)
* Build a TF-IDF vector index
* Save everything in `data/`

Options:

```bash
python setup.py --pdf report.pdf --chunk-size 400 --chunk-overlap 80
python setup.py --pdf report.pdf --rebuild
```

### 3. Run Web App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.


### 4. Using the Groq API

If you want your RAG system to generate answers using **Groq API**:

1. **Get an API key** from Groq.

2. **Set it in your terminal**:

   * **Windows (CMD / PowerShell)**:

     ```bash
     set GROQ_API_KEY=your_api_key_here
     ```
   * **Mac / Linux**:

     ```bash
     export GROQ_API_KEY=your_api_key_here
     ```

3. **Run the app** (`python app.py`).

* If the API key is set, questions you ask will be sent to the Groq API to generate answers.
* If the API key is not set, the system will still return **top relevant chunks** using TF-IDF only.


## ⚙️ How It Works

1. **Document Processing**

   * Loads PDF using `pypdf`
   * Cleans text and splits into overlapping chunks

2. **Vectorization & Retrieval**

   * Builds TF-IDF matrix with unigrams and bigrams
   * At query time: vectorizes user question → computes **cosine similarity** with all chunks
   * Returns top-K chunks with relevance scores

3. **Answer Generation (Groq API)**

   * Sends retrieved chunks to Groq API
   * API returns answers strictly from context
   * Falls back to **retrieval-only mode** if no API key


## 💡 Design Decisions

* **TF-IDF + LSA**: fast, lightweight, works without GPU
* **No LangChain / LlamaIndex**: everything transparent, simple, easy to audit
* **Anti-hallucination**: answers are strictly based on retrieved chunks; out-of-scope questions return “I could not find this information.”


## 📂 Project Structure

```
swiggy_rag/
├── document_processor.py   # PDF loading, cleaning, chunking
├── vector_store.py         # TF-IDF index + search
├── rag_engine.py           # Retrieval + Groq API call
├── setup.py                # PDF processing + index building
├── app.py                  # Web UI
├── requirements.txt
├── README.md
└── data/
    ├── chunks.json         # Extracted text chunks
    └── vector_store.pkl    # TF-IDF vector store
```

## 🧪 Example Questions

* "What was Swiggy's total revenue in FY2024?"
* "How many restaurant partners does Swiggy have?"
* "What are the key risks mentioned in the annual report?"
* "Who are the board of directors of Swiggy?"
* "What is Swiggy's market position in food delivery?"


