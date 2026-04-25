# LexAI: Advanced AI-Driven Legal Document Intelligence ⚖️🤖

![LexAI Logo](LexAI_Logo.png)

LexAI is a high-performance legal document analysis platform that leverages State-of-the-Art (SOTA) LLMs and Retrieval-Augmented Generation (RAG) to transform how legal professionals interact with complex contracts. Built with a focus on precision, security, and automation.

---

## 🚀 Key Features

- **Map-Reduce Summarization**: Effortlessly process 10,000+ word documents into structured executive summaries using LangChain's chunking and parallel processing.
- **RAG-Powered Q&A**: A hallucination-free chat interface grounded in your documents using FAISS vector indexing and LLaMA 3.1.
- **Automated Risk Assessment**: Multi-dimensional risk scoring (Compliance, Financial, Operational) powered by NLTK VADER and keyword frequency analysis.
- **Side-by-Side Comparison**: Semantic and line-level document diffing to identify critical changes between contract versions.
- **Compliance Classifier**: Automated regulatory domain identification across GDPR, HIPAA, IP, and 4 other key legal fields.

---

## 🛠️ Tech Stack

- **LLM Engine**: Groq (LLaMA 3.1 8B/70B)
- **Frameworks**: LangChain (LCEL), Streamlit
- **Vector Database**: FAISS
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **NLP Utilities**: NLTK, PyMuPDF, Scikit-learn
- **Automation**: FPDF

---

## 📐 System Architecture

### Information Flow
1. **Extraction**: Documents are parsed via PyMuPDF.
2. **Indexing**: Text is chunked (RecursiveCharacterTextSplitter) and stored in a FAISS vector index.
3. **Analysis**:
   - **Summarization**: Map-Reduce pipeline for long-form context.
   - **RAG**: Semantic retrieval for targeted querying.
   - **Risk**: Sentiment-based scoring and keyword analysis.
4. **Output**: Interactive Plotly visualizations and PDF exports.

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Vansh-Sharmaa/LexAI.git
   cd LexAI
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Create a `.env` file from `.env.example` and add your API keys:
   ```env
   GROQ_API_KEY=your_key_here
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## 🤝 Project Structure

```text
├── app.py                  # Main Streamlit Dashboard
├── summarization.py        # Map-Reduce & LCEL logic
├── rag.py                  # FAISS & Vector Retrieval
├── risk_analysis.py        # NLTK Risk Engine
├── comparison.py           # Document Diffing & Similarity
├── compliance.py           # Regex-based Classification
└── report_generation.py    # FPDF Reporting Logic
```

---

## 📜 License
This project is for educational and professional portfolio purposes.

**Developed by Vansh Sharma**  
[LinkedIn](https://linkedin.com/in/vansh-sharma) | [GitHub](https://github.com/Vansh-Sharmaa)
