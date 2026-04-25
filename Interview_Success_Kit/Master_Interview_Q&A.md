# LexAI — Interview Preparation Guide
### AI-Powered Legal Document Summarization & Risk Assessment

---

## 1. The 30-Second Elevator Pitch

> *"I built LexAI — an end-to-end AI-powered legal document intelligence platform. You upload any legal PDF and it automatically: summarizes the entire document using a Map-Reduce LLM pipeline over Groq's LLaMA 3.1, identifies risks with NLTK sentiment scoring, lets you query the document using natural language through a RAG pipeline backed by FAISS vector search, checks compliance across 7 regulatory frameworks like GDPR and HIPAA, compares two contracts side-by-side with semantic similarity scoring, and emails you a PDF report. The entire frontend is a premium Streamlit dashboard styled with custom CSS — Glassmorphism + Midnight Gold theme."*

---

## 2. Exact Tech Stack (What's Actually in the Code)

| Layer | Tech Used | Where |
|---|---|---|
| **LLM** | Groq API — LLaMA 3.1 8B Instant | `summarization.py`, `rag.py` |
| **LLM Orchestration** | LangChain LCEL (`prompt \| llm`) | `summarization.py`, `rag.py` |
| **Vector Search** | FAISS (Facebook AI Similarity Search) | `document_processing.py`, `rag.py` |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) | `document_processing.py`, `comparison.py` |
| **PDF Parsing** | PyMuPDF (`fitz`) | `document_processing.py` |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` | `document_processing.py`, `summarization.py` |
| **Sentiment/Risk** | NLTK VADER | `risk_analysis.py`, `utils.py` |
| **Compliance** | Regex keyword classification (local, no HTTP) | `compliance.py` |
| **Doc Comparison** | `difflib` + Sentence-Transformer cosine similarity | `comparison.py` |
| **PDF Reports** | FPDF | `report_generation.py` |
| **Email Delivery** | Resend API (sender: `onboarding@resend.dev`) | `report_generation.py` |
| **Frontend** | Streamlit + Custom CSS (Glassmorphism) | `app.py` |
| **Charts** | Plotly Express | `risk_analysis.py`, `app.py` |

---

## 3. Architecture — How Data Flows Through the System

```
User uploads PDF
       ↓
[PyMuPDF] → raw text
       ↓
[RecursiveCharacterTextSplitter] → chunks (1000 chars, 200 overlap)
       ↓                              ↓
[Sentence-Transformers]          [Map-Reduce Summarization]
[FAISS Index]                    Each chunk → Groq LLM (Map)
       ↓                         All summaries → Groq LLM (Reduce)
[RAG Q&A]                              ↓
User question →                  Final Summary
FAISS top-K retrieval →
Groq LLM answer

[NLTK VADER] → sentiment score per sentence → Risk Score (0–100)
[Regex matching] → Compliance categories (GDPR, HIPAA, Contracts, etc.)
[difflib + cosine sim] → Document comparison (if 2nd PDF uploaded)
[FPDF] → PDF report
[Resend API] → Email with PDF attachment
```

---

## 4. Core Concepts — Explain These in Any Interview

### 🔷 RAG (Retrieval-Augmented Generation)
**Problem it solves:** LLMs don't know your specific document. We "attach" the document.

**Step-by-step:**
1. Text is chunked into 1000-char segments with 200-char overlap
2. Each chunk is converted to a 384-dim embedding vector using `all-MiniLM-L6-v2`
3. All vectors are stored in a FAISS flat L2 index
4. When a user asks a question, the question is also embedded
5. FAISS finds the Top-3 closest chunks by cosine distance
6. Those chunks are injected as "context" into the Groq LLM prompt
7. The LLM answers **only from that context** — preventing hallucination

> **Key phrase for interview:** *"I grounded the LLM responses in the document using RAG so it cannot hallucinate facts that aren't in the contract."*

---

### 🔷 Map-Reduce Summarization
**Problem it solves:** LLMs have context windows (~8K tokens). A 50-page contract won't fit.

**Step-by-step:**
1. **Map Phase:** Each text chunk is individually sent to Groq LLM with the prompt: *"Summarize this legal extract in 3-5 sentences"*
2. **Reduce Phase:** All mini-summaries are combined and sent to Groq LLM again with: *"Combine these partial summaries into a well-structured executive summary"*
3. Result: A coherent, complete summary of a document of any length

**Implementation:** Uses LangChain LCEL — `ChatPromptTemplate | llm` — not the deprecated `LLMChain` pattern.

---

### 🔷 Sentiment-Based Risk Assessment
**How it works:**
- NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) scores the entire document
- The compound sentiment score captures how "negative/threatening" the language is
- Combined with keyword frequency for 3 risk categories: Compliance, Financial, Operational
- Each category has weighted scoring → total risk score 0–100
- Results rendered as Plotly pie chart + bar chart

**Severity levels:** Low / Medium / High / Critical

---

### 🔷 Document Comparison
**How it works:**
1. `difflib.Differ` performs line-by-line diff between the two cleaned documents
2. Common legal sections are extracted via regex patterns (Termination, Indemnification, etc.)
3. Sentence-Transformer embeddings compute **cosine similarity** per section
4. Output: Visual HTML diff (additions in green, deletions in red) + a Pandas DataFrame table

---

### 🔷 Compliance Classification
**How it works (important — no live web scraping):**
- 7 legal categories: GDPR, HIPAA, Contracts, IP, Employment, Real Estate, Tax Law
- Regex patterns match relevant keywords in the uploaded document
- Confidence score = keyword frequency / total word count × 1000
- Top 3 categories get their full compliance checklist and relevant regulations displayed
- **No external HTTP calls** — runs instantly and locally

---

## 5. Top Interview Questions & Answers

### Q1: "How did you handle long documents that exceed the LLM's context window?"
**Answer:** *"I implemented a Map-Reduce summarization pipeline. First, I split the document into 1000-character chunks with 200-character overlap using LangChain's `RecursiveCharacterTextSplitter`. Then in the Map phase, each chunk is individually summarized by Groq's LLaMA 3.1. In the Reduce phase, all those summaries are merged into a single executive summary. This lets me process documents of any length, regardless of the LLM's context window."*

---

### Q2: "Why FAISS over a cloud vector database like Pinecone?"
**Answer:** *"FAISS is a local, in-memory vector store by Facebook AI Research — it's extremely fast for similarity search (sub-millisecond for our scale), has zero infrastructure overhead, and needs no API keys or paid subscription. For a document analysis tool that processes one PDF at a time, FAISS is the right-sized tool. If this were a multi-user SaaS product storing millions of embeddings, I'd consider Pinecone or Weaviate."*

---

### Q3: "How do you prevent the LLM from making up legal information?"
**Answer:** *"I strictly grounded all responses using RAG. The system prompt explicitly says: 'Answer only using the provided context. If the answer is not in the context, say you don't know.' The LLM never reasons from its pre-training for Q&A — it only synthesizes information from the actual uploaded document."*

---

### Q4: "Why did you use Groq instead of OpenAI?"
**Answer:** *"Groq's inference engine is among the fastest available — LLaMA 3.1 8B on Groq runs at hundreds of tokens per second, which means document summarization completes in seconds. It's also free-tier friendly with a generous rate limit, making it ideal for a demo-ready resume project. The LangChain LCEL interface (`ChatPromptTemplate | llm`) makes it trivial to swap providers if needed."*

---

### Q5: "What was the hardest bug you fixed in this project?"
**Answer:** *"The biggest issue was infinite loading when a user uploaded a document. I diagnosed it and found the compliance module was making live HTTP requests to 14 external URLs (government sites, legal news) with retry loops and `time.sleep()` — all blocking the main thread. I completely replaced the web-scraping approach with a fast, offline keyword-classification engine using regex patterns. Document analysis time dropped from minutes to seconds."*

---

### Q6: "How does the document comparison work technically?"
**Answer:** *"Two things working together: First, Python's `difflib.Differ` performs character-level line comparison between the cleaned texts, highlighting additions in green and deletions in red. Second, for semantic similarity, I use the same Sentence-Transformer model from the RAG pipeline to embed each extracted section — like a Termination clause or Confidentiality section — and compute cosine similarity. This catches reworded-but-same-meaning clauses that a pure text diff would miss."*

---

### Q7: "Tell me about the email feature."
**Answer:** *"I used the Resend API to send emails with PDF attachments. The PDF is generated using FPDF — it includes the AI summary, risk scores, and compliance checklist. One challenge I hit: Resend's free tier doesn't allow sending from unverified domains, so you can't use a Gmail address as the sender. I solved this by using Resend's built-in verified sender domain (`onboarding@resend.dev`), which works without any custom domain setup on their free plan."*

---

## 6. Live Demo Strategy (Impress the Interviewer)

**Recommended order:**

1. **Start with Document Analysis (Tab 1)**
   - Upload a real legal PDF
   - Click "Begin Neural Analysis"
   - Show the loading steps: *"See — it's chunking, embedding, summarizing, scoring risks all in one click"*
   - Point to the Executive Summary card
   - **Click "Download Text Summary"** — show the .txt file downloads instantly

2. **Switch to Risk Dashboard (Tab 2)**
   - Show the Risk Score metric cards (Critical/High highlighted in red/orange)
   - Show the Plotly charts: *"This is a VADER sentiment-driven score — negative legal language raises the score"*

3. **Q&A Chat (Tab 3)**
   - Ask something very specific: *"What are the payment terms?"* or *"What happens if there is a breach?"*
   - Emphasize: *"It's answering from the actual document through FAISS vector search — not from the LLM's training data"*

4. **Report & Email (Tab 7)**
   - Generate the PDF report
   - Send email — show delivery message

---

## 7. Key Metrics to Quote

| Metric | Value |
|---|---|
| Document processing time | ~15–30 seconds |
| Embedding model dimensions | 384 (MiniLM-L6-v2) |
| FAISS top-K retrieval | Top 3 relevant chunks |
| Chunk size | 1000 characters, 200 overlap |
| Risk categories | 3 (Compliance, Financial, Operational) |
| Compliance domains | 7 (GDPR, HIPAA, Contracts, IP, Employment, Real Estate, Tax) |
| LLM model | LLaMA 3.1 8B Instant via Groq |
| UI tabs | 7 tabs in Streamlit |

---

## 8. Skills to Mention From This Project

- **NLP & LLMs:** LangChain LCEL, Groq API, LLaMA 3.1, Map-Reduce, RAG pipelines
- **Vector Databases:** FAISS, Sentence-Transformers, cosine similarity
- **Document Processing:** PyMuPDF, text chunking, difflib
- **Data Science:** NLTK VADER, Plotly, NumPy, Pandas
- **Backend:** Python, FPDF, Resend API integration
- **Frontend:** Streamlit, Custom CSS, Glassmorphism design
- **Software Engineering:** Modular architecture, error handling, session state management
