# 🧠 Data Science & AI Specific Q&A
### (Theoretical & Scenario-Based Questions)

Expect these questions in a technical "Data Science" interview round. These go beyond "what the app does" and focus on "how the AI works."

---

## 1. Vector Embeddings
**Q: "Why did you use `all-MiniLM-L6-v2` specifically?"**
A: *"It is a small, efficient model that produces 384-dimensional embeddings. For a project running on a standard laptop/server, the low latency is better than using larger models like BERT-base which are more computationally expensive but don't offer a massive increase in semantic accuracy for short chunks."*

**Q: "What is the difference between and 'Embedding' and a 'Hot-Encoded' vector?"**
A: *"One-Hot Encoding is sparse and doesn't capture relationships (every word is equidistant). Embeddings are dense, lower-dimensional, and capture semantic 'meaning' by placing similar words together in vector space based on the training data (distributional hypothesis)."*

---

## 2. Text Processing (NLP)
**Q: "Why is Chunk Overlap important?"**
A: *"If a critical sentence like 'The liability is limited to $1 Million' is cut in half by a chunking split, the RAG system might lose the context. Overlap (200 characters in our case) ensures that the end of one chunk and the start of the next share context, preventing data loss."*

**Q: "What are the limitations of NLTK VADER for legal text?"**
A: *"VADER is built for social media sentiment. Legal text is often 'Neutral' in tone but 'Negative' in consequence. I handled this by combining VADER with specific keyword weighting for legal terms like 'Breach', 'Termination', and 'Late' to make the risk score more accurate for this specific domain."*

---

## 3. RAG Architecture (Retrieval Augmented Generation)
**Q: "How do you handle 'Hallucinations' in the Chatbot?"**
A: *"I used strict ground-truth prompting. I instructed the LLM to 'Only answer from the provided context' and to 'Say I don't know if the answer is not present.' This limits the LLM's 'creative' generation and forces it to act as an information retriever."*

**Q: "What is the 'K' in Top-K retrieval, and how did you choose it?"**
A: *"The 'K' is the number of document chunks retrieved from FAISS. I chose K=3. Too small (K=1) might miss context; too large (K=10) might exceed the LLM's context window or include irrelevant noise that confuses the AI."*

---

## 4. Model Evaluation & Metrics
**Q: "As a Data Scientist, how would you evaluate the 'Summary' quality?"**
A: *"I would use **ROUGE scores** (Recall-Oriented Understudy for Gisting Evaluation) to compare the AI summary against a gold-standard human summary. I'd specifically look at ROUGE-L to see the longest common subsequence of words."*

---

## 5. Performance Optimization
**Q: "Why use Groq instead of just calling an API on your own computer?"**
A: *"Groq uses LPUs (Language Processing Units) which handle the 'sequential dependence' of generating one token after another much faster than a standard GPU. This makes the user experience nearly instant, which is critical for a production-grade application."*

---

## 6. Ethics & Bias
**Q: "Is there bias in your AI model?"**
A: *"Yes, all LLMs carry bias from their training data. In a legal context, this could mean the AI favors one party's language over another's. I mitigate this by focusing on 'extractive' summaries and strictly grounding chats in the user's specific document rather than general knowledge."*
