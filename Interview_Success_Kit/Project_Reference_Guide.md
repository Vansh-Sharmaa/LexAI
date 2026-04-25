# LexAI — Project Overview & Technical Reference
### AI-Powered Legal Document Summarization & Risk Assessment

> This document explains how the entire project works, module by module, in plain language. Use it to understand the codebase before an interview or demo.

---

## What Does This Project Do?

LexAI is a web application where you **upload a legal PDF** and get:
1. An AI-generated **executive summary** of the entire document
2. A **risk score** (0–100) with breakdown by severity and category
3. A **Q&A chatbot** where you can ask questions about the document
4. A **document comparison** tool (upload a second PDF and see differences)
5. A **compliance checklist** for the document's legal domain (GDPR, HIPAA, etc.)
6. **Legal update links** for the relevant regulatory domain
7. A **PDF report** you can download or email to yourself

---

## File Map — What Each File Does

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application — all UI, tabs, session state |
| `document_processing.py` | PDF text extraction (PyMuPDF) + chunking + FAISS index creation |
| `summarization.py` | Map-Reduce summarization using Groq LLM via LangChain LCEL |
| `rag.py` | RAG Q&A pipeline — FAISS retrieval + Groq LLM response |
| `risk_analysis.py` | NLTK VADER sentiment scoring + Plotly visualizations |
| `compliance.py` | Local regex-based compliance classification (7 domains, no HTTP) |
| `comparison.py` | difflib + cosine similarity document comparison |
| `report_generation.py` | FPDF PDF generation + Resend API email delivery |
| `utils.py` | Session state initialization, constants, shared utilities |
| `.env` | API keys (GROQ_API_KEY, RESEND_API_KEY, SENDER_EMAIL) |
| `requirements.txt` | All Python dependencies |

---

## How Each Feature Works (Step-by-Step)

### Feature 1: Document Analysis (Tab 1)

**When you click "Begin Neural Analysis":**

```
1. document_processing.extract_text_from_pdf(pdf)
   → PyMuPDF opens the PDF stream and concats text from all pages

2. document_processing.chunk_text(text)
   → RecursiveCharacterTextSplitter splits into ~1000-char chunks
   → 200-char overlap preserves context at chunk boundaries

3. document_processing.create_faiss_index(chunks)
   → Sentence-Transformers encodes all chunks to 384-dim vectors
   → FAISS IndexFlatL2 stores them for fast L2-distance search

4. summarization.generate_summary(text)
   → Map phase: each chunk → ChatGroq (LLaMA 3.1) → mini-summary
   → Reduce phase: all mini-summaries → ChatGroq → final summary
   → Uses LangChain LCEL: ChatPromptTemplate | llm

5. risk_analysis.advanced_risk_assessment(text)
   → NLTK VADER scores entire document (compound sentiment)
   → Keyword counts for 3 categories: Compliance, Financial, Operational
   → Combined into risk score 0–100

6. compliance.classify_document_type(text)
   → Regex matches GDPR/HIPAA/Contract/IP/Employment/Real Estate/Tax keywords
   → Returns top-3 categories with confidence scores

7. compliance.fetch_updates_for_document(text)
   → Uses top-2 categories, returns static update links (no web scraping)

8. compliance.fetch_document_compliance(text)
   → Returns detailed compliance checklist for each top category
```

---

### Feature 2: Q&A Chat (Tab 3)

```
User types question
   ↓
rag.generate_rag_response(question, faiss_index, chunks)
   ↓
Step 1: Embed the question → 384-dim vector (same SentenceTransformer)
Step 2: FAISS.search(query_vector, top_k=3) → 3 closest chunk indices
Step 3: Retrieve those 3 text chunks → join as "context"
Step 4: Build prompt:
   System: "Answer only from the context provided."
   Human: "Context: {context}\nQuestion: {question}"
Step 5: ChatGroq LLM generates answer
   ↓
Answer displayed in chat UI
```

**Why this works:** The LLM only sees the relevant parts of your document, so answers are always grounded in reality — no hallucination.

---

### Feature 3: Risk Dashboard (Tab 2)

```
risk_data = st.session_state.risk_data (computed during analysis)

Metrics displayed:
- Total Risk Score (0–100)
- Total Risks identified (keyword count)
- Critical count (from severity_counts dict)
- High count

Charts (rendered with Plotly):
- Pie chart: severity distribution (Low/Medium/High/Critical)
- Bar chart: score per risk category (Compliance/Financial/Operational)
```

**How the score is calculated:**
```python
total_score = min(100,
    sum(category_scores)           # keyword-weighted scores
    + (1 - sentiment_compound)*25  # negative sentiment raises score
    + min(30, avg_sentence_len*0.5) # complex sentences raise score
)
```

---

### Feature 4: Document Comparison (Tab 4)

```
You upload a second PDF
   ↓
compare_documents(original_text, new_text)
   ↓
Step 1: preprocess_text() — normalize whitespace, remove page numbers
Step 2: extract_document_sections() — regex extracts named legal sections
        (Termination, Confidentiality, Governing Law, etc.)
Step 3: calculate_statistics() — word count, sentence count, line diffs
Step 4: difflib.Differ.compare() — line-by-line diff of full documents
Step 5: For each matching section, cosine_similarity(embed1, embed2)
   ↓
Output 1: Color-coded HTML (additions=green, removals=red)
Output 2: Pandas DataFrame — section by section similarity + status
          (Identical / Minor Changes / Major Changes / Rewritten / Added / Removed)
```

---

### Feature 5: Compliance (Tab 5)

```
compliance_data = st.session_state.document_compliance

For each top category (up to 3):
- Confidence score (keyword_frequency × 1000)
- Requirements checklist (hardcoded, domain-specific)
- Relevant regulations list
- No HTTP requests — all local/instant
```

---

### Feature 6: Report & Email (Tab 7)

```
PDF Generation (FPDF):
- Summary text (latin-1 encoded, Unicode normalized)
- Risk scores and severity counts
- Compliance requirements per category
- Legal updates

Email (Resend API):
- Sender: onboarding@resend.dev (Resend's verified test domain)
- Attachment: PDF encoded as base64 string
- Free tier limitation: Can only deliver to your Resend-registered email
```

---

## Session State Keys (How App Remembers Data)

All these are initialized in `utils.initialize_session_state()`:

| Key | Type | Contains |
|---|---|---|
| `document_processed` | bool | Whether analysis is complete |
| `full_text` | str | Raw PDF text |
| `text_chunks` | list[str] | Split chunks |
| `faiss_index` | faiss.Index | Vector search index |
| `summaries['document']` | str | AI-generated summary |
| `risk_data` | dict | Risk scores, counts, categories |
| `document_categories` | list | Top classification results |
| `legal_updates` | dict | Legal update links per category |
| `document_compliance` | dict | Compliance checklist per category |
| `chat_history` | list | (role, message) pairs |
| `pdf_buffer` | BytesIO | Generated PDF data |
| `comparison_report` | BytesIO | HTML comparison report |

---

## Common Questions About the Code

### "Why `--server.fileWatcherType none` when running?"
PyTorch (installed as a sentence-transformers dependency) causes a `RuntimeError` with Streamlit's file watcher. This flag disables the file watcher and avoids the conflict.

### "Why is `langchain-groq` used instead of `langchain-openai`?"
Groq provides ultra-fast inference for open-source models (LLaMA 3.1) at no cost. The interface is identical through LangChain — you could switch to OpenAI by just changing 1 line.

### "Why is compliance done locally now, not via web scraping?"
The original BeautifulSoup scraper was making 14 HTTP requests with retry loops and `time.sleep()` — causing the app to hang for 2–5 minutes during document analysis. The local keyword regex classifier runs in milliseconds with identical UX quality.

### "What's LCEL (LangChain Expression Language)?"
It's the modern LangChain API: `chain = prompt | llm`. You call `chain.invoke({"var": value})`. It replaces the deprecated `LLMChain.run()` pattern which could silently hang on newer versions.

---

## Project Run Command

```powershell
# From the project directory:
.\venv\Scripts\python -m streamlit run app.py --server.fileWatcherType none
```

**Access at:** http://localhost:8501

---

## Environment Variables Required

```env
GROQ_API_KEY=your_groq_api_key_here
RESEND_API_KEY=your_resend_api_key_here
SENDER_EMAIL=your_email@gmail.com   # Not used for sending, kept for reference
```
