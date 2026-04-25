# 🤖 How to use other AI (ChatGPT/Gemini) as your Tutor

If you want to practice your interview with another AI (like ChatGPT, Gemini, or Claude), you don't need to upload your code. You just need to give it the **"Master Context"** so it understands exactly how your project works.

---

## 📝 The "Copy-Paste" Master Prompt

**Copy the entire block below and paste it into ChatGPT/Gemini:**

---

> **ACT AS MY TECHNICAL INTERVIEWER & AI TUTOR**
> 
> I am an MCA (AI & Data Science) student. This is my project: **LexAI — Legal Document Intelligence Platform**. 
> 
> **PROJECT ARCHITECTURE:**
> 1. **Tech Stack:** Python, Streamlit, LangChain (LCEL), Groq (LLaMA 3.1 8B), FAISS, Sentence-Transformers (all-MiniLM-L6-v2), PyMuPDF, NLTK VADER.
> 2. **RAG Pipeline:** PDF text extracted via PyMuPDF → Chunked via RecursiveCharacterTextSplitter (1000 chars, 200 overlap) → Embedded (384-dim) → Stored in FAISS Flat-L2 Index → Top-K retrieval (K=3) for grounded Q&A.
> 3. **Summarization:** Implemented a Map-Reduce pipeline to handle large documents. Map phase summarizes chunks sequentially; Reduce phase synthesizes them into a final Executive Summary.
> 4. **Risk Assessment:** Hybrid engine. Uses NLTK VADER sentiment analysis + legal keyword frequency to score documents (0-100) across Compliance, Financial, and Operational risks.
> 5. **Comparison:** Comparison of two PDFs using `difflib` for text diffs and Cosine Similarity between embeddings for semantic clause matching.
> 6. **Compliance:** Fast, offline regex-based classifier for 7 domains (GDPR, HIPAA, etc.) — replacing slow web scraping.
> 
> **YOUR TASKS:**
> 1. Act as a **Senior AI Engineer** and interview me. Ask me one deep technical question at a time.
> 2. If I answer correctly, challenge me with a harder follow-up.
> 3. If I get it wrong, explain the concept to me in simple terms so I learn.
> 4. Focus on RAG, Vector Space, Map-Reduce, and Python implementation.
> 
> Let's start the interview! Introduce yourself as my interviewer and ask the first question.

---

## 💡 Why this is beneficial:

1.  **Mock Interviews**: The other AI will grill you on things like "How do you handle vector collisions?" or "Why 200-character overlap?". This is exactly what real interviewers do.
2.  **Concept Clarification**: If you don't understand a part of your code (like the `faiss_index.search()` part), you can ask: *"Hey, explain line 88 of my RAG pipeline to me like I'm 5."* 
3.  **Brainstorming**: You can ask: *"What are 3 ways I could improve the accuracy of my legal risk scoring?"*
4.  **No Code Required**: Since I've given you the "Context" in the prompt above, the AI already knows the logic without seeing the actual `.py` files.

**I have added this file to your `Interview_Success_Kit` folder!** 🚀
