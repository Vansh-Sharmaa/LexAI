# Resume Content — LexAI: Legal Document Intelligence

## Resume Bullet Points

### Option A — Concise (3 bullets, for tight resume space)

**LexAI: AI-Powered Legal Document Summarization & Risk Assessment**
*LangChain · Groq LLM (LLaMA 3.1) · FAISS · Sentence-Transformers · Streamlit · Plotly · NLTK · Resend API*

- Built an end-to-end legal AI platform with a **Map-Reduce summarization pipeline** using **LangChain LCEL** and **Groq LLaMA 3.1**, processing 10K+ word contracts into structured executive summaries via a chunk→map→reduce architecture.

- Developed a **Retrieval-Augmented Generation (RAG) Q&A system** using **FAISS vector indexing** and **Sentence-Transformers (MiniLM-L6-v2)**, enabling hallucination-free natural language querying grounded exclusively in uploaded document content.

- Engineered an **NLTK VADER sentiment-based risk scoring engine** (Low/High/Critical), keyword-driven compliance classification across 7 regulatory domains (GDPR, HIPAA, Contracts, IP, Employment, Real Estate, Tax), and automated PDF/email delivery via **Resend API** — all served through a premium **Glassmorphism Streamlit dashboard**.

---

### Option B — Detailed (5 bullets, for project section)

**LexAI: AI-Powered Legal Document Intelligence Platform**
*LangChain · Groq API · LLaMA 3.1 · FAISS · Sentence-Transformers · Streamlit · PyMuPDF · NLTK · Plotly · Resend API*

- Architected a **Map-Reduce LLM pipeline** using **LangChain LCEL** (`ChatPromptTemplate | llm`) and **Groq LLaMA 3.1 8B**, splitting documents with `RecursiveCharacterTextSplitter` (1000-char chunks, 200-char overlap) and running parallel summarization before a final reduce step — enabling unlimited document length processing.

- Built a **RAG-based Q&A pipeline**: extracted text with **PyMuPDF**, embedded chunks into 384-dimensional vectors using **Sentence-Transformers (all-MiniLM-L6-v2)**, stored in a **FAISS Flat-L2 index**, and retrieved Top-3 semantically similar chunks to ground every LLM response — eliminating hallucination.

- Designed an **automated risk assessment engine** using **NLTK VADER** sentiment scoring combined with keyword frequency analysis across 3 risk categories (Compliance, Financial, Operational), producing a composite 0–100 risk score visualized with **Plotly** pie and bar charts.

- Implemented **multi-document comparison** using Python `difflib` for line-level diff (additions/deletions in color-coded HTML) plus **cosine similarity** between Sentence-Transformer embeddings of extracted legal sections (Termination, Indemnification, Governing Law, etc.) for semantic drift detection.

- Built a **local compliance classifier** using regex pattern matching across 7 legal domains, eliminating the latency of live web scraping; integrated **FPDF** for PDF report generation and **Resend API** for automated email delivery with base64-encoded attachments — all deployed as a 7-tab **Streamlit** dashboard with a premium custom CSS interface.

---

## Tech Stack Summary (for Skills Section)

| Category | Technologies |
|---|---|
| **LLM / AI** | LangChain LCEL, Groq API, LLaMA 3.1 8B, Map-Reduce, RAG |
| **NLP** | NLTK VADER Sentiment Analysis, Sentence-Transformers (MiniLM) |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Document Processing** | PyMuPDF, RecursiveCharacterTextSplitter, difflib |
| **Frontend** | Streamlit, Plotly, Custom CSS (Glassmorphism) |
| **Backend / APIs** | Python, FPDF, Resend API, Regex pattern classification |
| **Architecture** | RAG Pipeline, Map-Reduce Summarization, Modular Python |

---

## Updated Architecture — Mermaid Diagram Code

Paste into [mermaid.live](https://mermaid.live/) to generate a clean diagram image.

### Detailed Architecture Diagram

```mermaid
graph TB
    subgraph Input["📄 Input Layer"]
        A["PDF Upload"] --> B["PyMuPDF Text Extraction"]
    end

    subgraph Processing["⚙️ Processing Pipeline"]
        B --> C["RecursiveCharacterTextSplitter\n1000 chars · 200 overlap"]
        C --> D["Sentence-Transformers\nall-MiniLM-L6-v2 · 384-dim"]
        D --> E["FAISS Flat-L2 Index"]
    end

    subgraph AI["🤖 AI Analysis Engine"]
        C --> F["Map Phase\nChunk → Groq LLaMA 3.1"]
        F --> G["Reduce Phase\nAll summaries → Groq LLaMA 3.1"]
        G --> H["Executive Summary"]
        E --> I["Top-3 Chunk Retrieval"]
        I --> J["RAG Q&A Response\nGroq LLaMA 3.1"]
        C --> K["NLTK VADER Sentiment\n+ Keyword Frequency"]
        K --> L["Risk Score 0-100\nCompliance / Financial / Operational"]
    end

    subgraph Compliance["📜 Compliance Module"]
        B --> M["Regex Keyword Classifier\nLocal · No HTTP calls"]
        M --> N["Top-3 Legal Domains\nGDPR / HIPAA / Contracts\nIP / Employment / Real Estate / Tax"]
        N --> O["Compliance Checklist\n+ Relevant Regulations"]
    end

    subgraph Comparison["🔀 Document Comparison"]
        B --> P["difflib.Differ\nLine-level diff"]
        P --> Q["Color-coded HTML\nGreen=Added · Red=Removed"]
        B --> R["Cosine Similarity\nper Legal Section"]
        R --> S["Tabular Comparison\nPandas DataFrame"]
    end

    subgraph Output["📊 Output Layer"]
        H --> T["PDF Report\nFPDF"]
        L --> T
        O --> T
        T --> U["Email Delivery\nResend API\nonboarding@resend.dev"]
        H --> V["Download .txt Summary"]
        L --> W["Plotly Charts\nPie + Bar"]
    end

    subgraph Frontend["🖥️ Streamlit Dashboard · 7 Tabs"]
        X1["Document Analysis"]
        X2["Risk Dashboard"]
        X3["Q&A Chat"]
        X4["Comparison"]
        X5["Compliance"]
        X6["Legal Updates"]
        X7["Report & Email"]
    end

    style Input fill:#1a1a2e,stroke:#d4af37,color:#fff
    style Processing fill:#16213e,stroke:#d4af37,color:#fff
    style AI fill:#0f3460,stroke:#d4af37,color:#fff
    style Compliance fill:#1a1a2e,stroke:#8b5cf6,color:#fff
    style Comparison fill:#16213e,stroke:#8b5cf6,color:#fff
    style Output fill:#0f3460,stroke:#d4af37,color:#fff
    style Frontend fill:#050608,stroke:#d4af37,color:#fff
```

### Simple Flowchart (for presentations)

```mermaid
flowchart LR
    A["📄 PDF Upload"] --> B["Text Extraction\nPyMuPDF"]
    B --> C["Chunking\n1000 chars"]
    C --> D["FAISS\nVector Index"]
    D --> E["RAG Q&A\nGroq LLaMA 3.1"]
    C --> F["Map-Reduce\nSummary"]
    C --> G["VADER\nRisk Score"]
    B --> H["Regex\nCompliance"]
    B --> I["difflib\nDoc Comparison"]
    E & F & G & H & I --> J["📋 PDF Report\nFPDF"]
    J --> K["📧 Email\nResend API"]
```
