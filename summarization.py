from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


@st.cache_resource
def load_llm():
    try:
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_retries=2,
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None


llm = load_llm()


def _chunk_text(text: str, chunk_size: int = 6000) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
    )
    return splitter.split_text(text)


def generate_summary(text: str) -> str:
    """
    Summarise a legal document using a simple map→reduce approach
    that works with the modern LangChain invoke() API.
    Falls back to a safe extractive summary if the LLM is unavailable.
    """
    if not text:
        return "No content to summarise."

    if llm is None:
        # Extractive fallback – first 1 500 chars
        return (
            "⚠️ LLM unavailable – extractive preview:\n\n"
            + text[:1500].strip()
            + "\n\n[Install / configure the GROQ_API_KEY to enable AI summaries]"
        )

    try:
        chunks = _chunk_text(text)

        # ── Map phase: summarise each chunk individually ──────────────────────
        map_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a legal analyst. Summarise the following extract "
                    "from a legal document concisely (3-5 sentences).",
                ),
                ("human", "{chunk}"),
            ]
        )
        map_chain = map_prompt | llm

        chunk_summaries: list[str] = []
        for chunk in chunks:
            response = map_chain.invoke({"chunk": chunk})
            chunk_summaries.append(
                response.content if hasattr(response, "content") else str(response)
            )

        # ── Reduce phase: merge chunk summaries ───────────────────────────────
        combined = "\n\n".join(chunk_summaries)

        # If combined fits in one call, do a single reduce
        reduce_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a legal analyst. Combine the following partial summaries "
                    "into a single, well-structured executive summary of the legal "
                    "document. Use clear paragraphs covering: purpose, key obligations, "
                    "notable risks, and recommendations.",
                ),
                ("human", "{summaries}"),
            ]
        )
        reduce_chain = reduce_prompt | llm

        # Trim combined if too long to avoid context-window errors
        if len(combined) > 12_000:
            combined = combined[:12_000]

        final = reduce_chain.invoke({"summaries": combined})
        return final.content if hasattr(final, "content") else str(final)

    except Exception as e:
        return f"⚠️ Summary generation failed: {str(e)}\n\nDocument preview:\n{text[:800].strip()}"