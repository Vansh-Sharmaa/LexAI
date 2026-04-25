import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import torch
try:
    torch.classes.__path__ = []
except Exception:
    pass

import plotly.express as px
import nltk
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'vader_lexicon'])
import streamlit as st
import os
import base64
import faiss
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
from fpdf import FPDF
from io import BytesIO
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
from typing import Tuple, List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import difflib
import requests
from bs4 import BeautifulSoup
import base64

import nltk
import os
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LexAI · Legal Document Intelligence",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    nltk.download('vader_lexicon', download_dir=os.path.join(os.getcwd(), "nltk_data"))
except LookupError as e:
    st.error(f"Error downloading NLTK resources: {e}.")
    st.stop()

from nltk.sentiment import SentimentIntensityAnalyzer

from document_processing import extract_text_from_pdf
from document_processing import chunk_text, create_faiss_index
from rag import generate_rag_response
from risk_analysis import advanced_risk_assessment, visualize_risks
from summarization import generate_summary
from comparison import compare_documents, export_comparison_report, compare_documents_tabular
from compliance import fetch_updates_for_document, classify_document_type, fetch_document_compliance
from report_generation import generate_pdf, send_email, create_email_text
from utils import initialize_session_state

initialize_session_state()

# ─────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────
def validate_email(email):
    import re
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:ital,wght@0,700;0,900;1,700&display=swap');

:root {
    --primary: #d4af37;
    --primary-glow: rgba(212, 175, 55, 0.3);
    --bg-dark: #050608;
    --card-bg: rgba(255, 255, 255, 0.03);
    --card-border: rgba(255, 255, 255, 0.08);
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: var(--bg-dark) !important;
    color: var(--text-main);
    font-family: 'Outfit', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 5rem !important; max-width: 1400px !important; }

/* ── Mesh Gradient Background ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: -2;
    background: 
        radial-gradient(circle at 0% 0%, rgba(212, 175, 55, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 100% 0%, rgba(139, 92, 246, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 50% 100%, rgba(212, 175, 55, 0.05) 0%, transparent 50%),
        var(--bg-dark);
}

.stApp::after {
    content: '';
    position: fixed; inset: 0; z-index: -1;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
    opacity: 0.04;
    pointer-events: none;
}

/* ── Custom Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: rgba(212, 175, 55, 0.2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(212, 175, 55, 0.4); }

/* ── Masthead ── */
.masthead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 2.5rem 0;
    margin-bottom: 3rem;
    border-bottom: 1px solid rgba(212, 175, 55, 0.15);
    position: relative;
    animation: fadeInDown 0.8s ease-out;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.masthead-left { display: flex; align-items: center; gap: 20px; }
.masthead-emblem {
    width: 56px; height: 56px;
    border: 1px solid var(--primary);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    color: var(--primary);
    background: rgba(212, 175, 55, 0.05);
    box-shadow: 0 0 30px var(--primary-glow);
    position: relative;
    overflow: hidden;
}

.masthead-emblem::after {
    content: '';
    position: absolute; inset: -50%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
    transform: rotate(45deg);
    animation: shine 3s infinite;
}

@keyframes shine {
    0% { left: -100%; }
    100% { left: 100%; }
}

.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: #fff;
    letter-spacing: -0.5px;
    line-height: 1;
    margin: 0;
}
.masthead-sub {
    font-size: 0.85rem;
    color: var(--text-muted);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 6px;
    font-weight: 500;
}
.masthead-badge {
    font-family: 'JetBrains+Mono', monospace;
    font-size: 0.75rem;
    color: var(--primary);
    border: 1px solid rgba(212, 175, 55, 0.3);
    padding: 6px 16px;
    border-radius: 100px;
    letter-spacing: 1px;
    text-transform: uppercase;
    background: rgba(212, 175, 55, 0.05);
    backdrop-filter: blur(10px);
}

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    gap: 8px !important;
    padding-bottom: 4px !important;
}
div[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    padding: 12px 24px !important;
    border-radius: 8px 8px 0 0 !important;
    border: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: transparent !important;
}
div[data-baseweb="tab"]:hover { color: #fff !important; background: rgba(255, 255, 255, 0.03) !important; }
div[aria-selected="true"][data-baseweb="tab"] {
    color: var(--primary) !important;
    background: rgba(212, 175, 55, 0.05) !important;
    position: relative;
}
div[aria-selected="true"][data-baseweb="tab"]::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 2px; background: var(--primary);
    box-shadow: 0 0 10px var(--primary);
}

/* ── Section Header ── */
.sec-header { display: flex; align-items: center; gap: 16px; margin: 3rem 0 2rem; }
.sec-header-num { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--primary); letter-spacing: 2px; }
.sec-header h3 { font-family: 'Playfair Display', serif !important; font-size: 1.6rem !important; font-weight: 700 !important; color: #fff !important; margin: 0 !important; }
.sec-divider { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(212, 175, 55, 0.3), transparent); }

/* ── Metric Cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; }
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 28px;
    backdrop-filter: blur(12px);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-5px);
    border-color: rgba(212, 175, 55, 0.3);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}
.metric-card-label { 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 0.7rem; 
    letter-spacing: 2px; 
    text-transform: uppercase; 
    color: var(--text-muted); 
    margin-bottom: 12px; 
}
.metric-card-value { 
    font-family: 'Playfair Display', serif; 
    font-size: 3.2rem; 
    font-weight: 900; 
    line-height: 1; 
}
.metric-card-value.gold { color: var(--primary); text-shadow: 0 0 20px var(--primary-glow); }
.metric-card-value.neutral { color: #fff; }
.metric-card-value.danger { color: #ff4b4b; text-shadow: 0 0 20px rgba(255, 75, 75, 0.3); }
.metric-card-value.warn { color: #ffa41b; text-shadow: 0 0 20px rgba(255, 164, 27, 0.3); }
.metric-card-sub { font-size: 0.8rem; color: var(--text-muted); margin-top: 8px; }

/* ── Hero / Landing ── */
.hero-container {
    text-align: center;
    padding: 4rem 0;
    max-width: 800px;
    margin: 0 auto;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 900;
    color: #fff;
    margin-bottom: 1.5rem;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.25rem;
    color: var(--text-muted);
    margin-bottom: 3rem;
    line-height: 1.6;
}

/* ── Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 2px dashed rgba(212, 175, 55, 0.2) !important;
    border-radius: 20px !important;
    padding: 40px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(212, 175, 55, 0.03) !important;
    border-color: var(--primary) !important;
}
.upload-hint { 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 0.75rem; 
    color: var(--text-muted); 
    letter-spacing: 1px; 
    text-transform: uppercase; 
    margin-top: 12px; 
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.1), transparent) !important;
    color: var(--primary) !important;
    border: 1px solid rgba(212, 175, 55, 0.4) !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
    background: var(--primary) !important;
    color: #000 !important;
    border-color: var(--primary) !important;
    box-shadow: 0 0 30px var(--primary-glow) !important;
    transform: translateY(-2px);
}
.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: #000 !important;
}

/* ── Chat ── */
[data-testid="stChatMessageContent"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    backdrop-filter: blur(10px);
}
.stChatMessage { margin-bottom: 20px !important; }

/* ── Alert ── */
.stAlert {
    background: rgba(212, 175, 55, 0.05) !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(5px);
}

/* ── Summary Card ── */
.summary-card {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.05), transparent);
    border: 1px solid rgba(212, 175, 55, 0.15);
    border-radius: 20px;
    padding: 40px;
    font-size: 1.1rem;
    line-height: 1.8;
    color: var(--text-main);
    position: relative;
}
.summary-card::before {
    content: '“';
    font-family: 'Playfair Display', serif;
    font-size: 8rem;
    color: rgba(212, 175, 55, 0.1);
    position: absolute;
    top: -20px; left: 10px;
    line-height: 1;
}

/* ── Classification Badges ── */
.class-badge { 
    display: flex; align-items: center; justify-content: space-between; 
    background: rgba(255, 255, 255, 0.02); 
    border: 1px solid var(--card-border); 
    border-radius: 12px; 
    padding: 14px 20px; 
    margin-bottom: 12px; 
    backdrop-filter: blur(5px);
}
.class-name { font-weight: 600; font-size: 0.95rem; color: #fff; }
.class-conf { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--primary); }

.update-card { 
    background: rgba(255, 255, 255, 0.02); 
    border: 1px solid var(--card-border); 
    border-left: 4px solid var(--primary); 
    border-radius: 12px; 
    padding: 20px; 
    margin-bottom: 16px; 
}
.update-title { font-weight: 600; font-size: 1rem; color: #fff; margin-bottom: 6px; }
.update-source { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
.update-snippet { font-size: 0.9rem; color: var(--text-muted); margin-top: 10px; line-height: 1.6; }

.regulation-item { 
    display: inline-block; 
    background: rgba(212, 175, 55, 0.1); 
    border: 1px solid rgba(212, 175, 55, 0.3); 
    color: var(--primary); 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 0.75rem; 
    padding: 5px 12px; 
    border-radius: 8px; 
    margin: 4px; 
}
.compliance-item { 
    padding: 16px 20px; 
    border-left: 3px solid rgba(212, 175, 55, 0.4); 
    margin-bottom: 12px; 
    font-size: 0.95rem; 
    color: var(--text-muted); 
    line-height: 1.6; 
    background: rgba(255,255,255,0.01);
    border-radius: 0 12px 12px 0;
}

/* ── Footer ── */
.lex-footer {
    margin-top: 6rem;
    padding: 3rem 0;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}
.lex-footer-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
}
.lex-footer-info {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}
</style>
"""


def sec_header(num, title):
    st.markdown(f"""
    <div class="sec-header">
        <span class="sec-header-num">{num}</span>
        <h3>{title}</h3>
        <div class="sec-divider"></div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Masthead ──
    st.markdown("""
    <div class="masthead">
        <div class="masthead-left">
            <div class="masthead-emblem">⚖</div>
            <div>
                <div class="masthead-title">LexAI</div>
                <div class="masthead-sub">Legal Document Intelligence · Vansh Sharma</div>
            </div>
        </div>
        <div class="masthead-badge">AI-Powered · Premium v2.1</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.document_processed:
        # ── Hero Section (Only visible when no document is processed) ──
        st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">Intelligent Legal Document Analysis</h1>
            <p class="hero-subtitle">
                Automate contract review, identify hidden risks, and ensure regulatory compliance 
                with our advanced AI-driven RAG pipeline.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Document Analysis",
        "Risk Dashboard",
        "Q&A Chat",
        "Comparison",
        "Compliance",
        "Legal Updates",
        "Report & Email",
    ])

    # ═══════════════════ TAB 1 ═══════════════════
    with tab1:
        sec_header("01", "Document Analysis")

        uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type=["pdf"])
        st.markdown('<p class="upload-hint">Supported: PDF · All document types</p>', unsafe_allow_html=True)

        if uploaded_file and not st.session_state.document_processed:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("▶  Begin Neural Analysis", type="primary", use_container_width=True):
                with st.status("Initializing Analysis Engine…", expanded=True) as status:
                    try:
                        st.write("🌌 Establishing neural context…")
                        st.session_state.full_text = extract_text_from_pdf(uploaded_file)
                        
                        st.write("🧩 Segmenting document architecture…")
                        st.session_state.text_chunks = chunk_text(st.session_state.full_text)
                        
                        st.write("🔍 Indexing vector space…")
                        st.session_state.faiss_index = create_faiss_index(st.session_state.text_chunks)
                        
                        st.write("📝 Distilling executive intelligence…")
                        st.session_state.summaries['document'] = generate_summary(st.session_state.full_text)
                        
                        st.write("⚖ Calculating risk vectors…")
                        st.session_state.risk_data = advanced_risk_assessment(st.session_state.full_text)
                        
                        st.write("🏷 Categorising legal domain…")
                        st.session_state.document_categories = classify_document_type(st.session_state.full_text)
                        
                        st.write("🌍 Synchronising global legal updates…")
                        st.session_state.legal_updates = fetch_updates_for_document(st.session_state.full_text)
                        
                        st.write("🛡 Verifying compliance matrix…")
                        st.session_state.document_compliance = fetch_document_compliance(st.session_state.full_text)
                        
                        status.update(label="Analysis complete. All intelligence nodes synchronized.", state="complete", expanded=False)
                        st.success("Analysis finalized successfully.")
                        st.session_state.document_processed = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
                        st.session_state.document_processed = False

        if st.session_state.document_processed:
            sec_header("02", "Executive Summary")
            st.markdown(f'<div class="summary-card">{st.session_state.summaries.get("document","No summary available.")}</div>', unsafe_allow_html=True)

            if st.session_state.document_categories:
                st.markdown("<br>", unsafe_allow_html=True)
                sec_header("03", "Document Classification")
                for category, confidence in st.session_state.document_categories[:3]:
                    st.markdown(f'<div class="class-badge"><span class="class-name">{category}</span><span class="class-conf">{confidence:.2f} confidence</span></div>', unsafe_allow_html=True)
                    st.progress(min(1.0, float(confidence) / 100.0))

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                summary_text = st.session_state.summaries.get('document', "")
                st.download_button(
                    "↓  Download Text Summary",
                    data=summary_text.encode("utf-8"),
                    file_name="document_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with c2:
                if st.button("⊞  Generate Full PDF Report", use_container_width=True):
                    with st.spinner("Compiling…"):
                        pdf_buffer = generate_pdf(st.session_state.summaries['document'], st.session_state.risk_data, getattr(st.session_state,'legal_updates',None), getattr(st.session_state,'document_compliance',None))
                        st.session_state.pdf_buffer = pdf_buffer
                        st.success("Report ready in the Report & Email tab.")
        elif not uploaded_file:
            st.info("Upload a PDF document above to begin analysis.")

    # ═══════════════════ TAB 2 ═══════════════════
    with tab2:
        sec_header("01", "Risk Overview")
        if st.session_state.document_processed:
            risk_data = st.session_state.risk_data
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card gold">
                    <div class="metric-card-label">Overall Score</div>
                    <div class="metric-card-value gold">{risk_data.get("total_score",0)}</div>
                    <div class="metric-card-sub">out of 100</div>
                </div>
                <div class="metric-card neutral">
                    <div class="metric-card-label">Total Risks</div>
                    <div class="metric-card-value neutral">{risk_data.get("total_risks",0)}</div>
                    <div class="metric-card-sub">identified</div>
                </div>
                <div class="metric-card danger">
                    <div class="metric-card-label">Critical</div>
                    <div class="metric-card-value danger">{risk_data["severity_counts"].get("Critical",0)}</div>
                    <div class="metric-card-sub">immediate action</div>
                </div>
                <div class="metric-card warn">
                    <div class="metric-card-label">High</div>
                    <div class="metric-card-value warn">{risk_data["severity_counts"].get("High",0)}</div>
                    <div class="metric-card-sub">elevated level</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            sec_header("02", "Risk Visualisations")
            fig1, fig2 = visualize_risks(risk_data)
            if fig1 and fig2:
                for fig in (fig1, fig2):
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        font=dict(color='#94a3b8', family='Outfit'),
                        margin=dict(t=40, b=0, l=0, r=0)
                    )
                cv1, cv2 = st.columns(2)
                with cv1: st.plotly_chart(fig1, use_container_width=True)
                with cv2: st.plotly_chart(fig2, use_container_width=True)

            sec_header("03", "Category Breakdown")
            if risk_data.get('categories'):
                df = pd.DataFrame.from_dict(risk_data['categories'], orient='index')
                st.dataframe(df, column_config={"score": st.column_config.ProgressColumn("Score", format="%f", min_value=0, max_value=40)}, use_container_width=True)
        else:
            st.info("Upload and analyse a document to view the risk dashboard.")

    # ═══════════════════ TAB 3 ═══════════════════
    with tab3:
        sec_header("01", "Document Q&A")
        if st.session_state.document_processed:
            for role, msg in st.session_state.chat_history:
                with st.chat_message(role): st.write(msg)
            query = st.chat_input("Ask a question about the document…")
            if query:
                with st.spinner("Analysing…"):
                    response = generate_rag_response(query, st.session_state.faiss_index, st.session_state.text_chunks)
                    st.session_state.chat_history.extend([("user", query), ("assistant", response)])
                    st.rerun()
        else:
            st.info("Upload and analyse a document to activate Q&A.")
            with st.expander("Explore sample intelligence queries"):
                st.markdown("""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div class="stAlert" style="padding: 15px;">What are the key obligations in this contract?</div>
                    <div class="stAlert" style="padding: 15px;">Explain the termination clause in plain language.</div>
                    <div class="stAlert" style="padding: 15px;">What are the payment terms and penalties?</div>
                    <div class="stAlert" style="padding: 15px;">Are there any concerning liability clauses?</div>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════ TAB 4 ═══════════════════
    with tab4:
        sec_header("01", "Document Comparison")
        if st.session_state.document_processed:
            compare_file = st.file_uploader("Upload Comparison Document (PDF)", type=["pdf"])
            st.markdown('<p class="upload-hint">Compared against your original document</p>', unsafe_allow_html=True)
            if compare_file:
                try:
                    compare_text = extract_text_from_pdf(compare_file)
                    comparison = compare_documents(st.session_state.full_text, compare_text)
                    comparison_table = compare_documents_tabular(st.session_state.full_text, compare_text)

                    sec_header("02", "Visual Differential")
                    st.markdown(f'<div style="background:var(--card-bg);border:1px solid var(--card-border);border-radius:16px;padding:30px;font-family:JetBrains Mono,monospace;font-size:0.85rem;line-height:1.7;backdrop-filter:blur(10px);">{comparison}</div>', unsafe_allow_html=True)

                    sec_header("03", "Tabular Comparison")
                    st.dataframe(comparison_table, use_container_width=True)

                    ce1, ce2 = st.columns(2)
                    with ce1:
                        if st.button("⊞  Generate Comparison Report", use_container_width=True):
                            with st.spinner("Building report…"):
                                report_buffer = export_comparison_report(st.session_state.full_text, compare_text, "Original Document", compare_file.name)
                                st.session_state.comparison_report = report_buffer
                                st.success("Report ready.")
                    with ce2:
                        if st.session_state.get('comparison_report'):
                            st.download_button("↓  Download Comparison Report", data=st.session_state.comparison_report, file_name="Document_Comparison_Report.html", mime="text/html", use_container_width=True)
                except Exception as e:
                    st.error(f"Comparison failed: {str(e)}")
            else:
                st.info("Upload a second PDF to compare against your original.")
        else:
            st.info("Upload and analyse a document first.")

    # ═══════════════════ TAB 5 ═══════════════════
    with tab5:
        sec_header("01", "Compliance Requirements")
        if st.session_state.document_processed and hasattr(st.session_state, 'document_compliance'):
            compliance_data = st.session_state.document_compliance
            if not compliance_data:
                st.info("No specific compliance requirements identified.")
            else:
                for category, data in compliance_data.items():
                    confidence = data.get('confidence', 0)
                    with st.expander(f"{category}  ·  {confidence:.2f} confidence"):
                        st.markdown("**Requirements**")
                        for item in data.get('requirements', []):
                            st.markdown(f'<div class="compliance-item">{item}</div>', unsafe_allow_html=True)
                        st.markdown("<br>**Relevant Regulations**")
                        st.markdown("".join(f'<span class="regulation-item">{r}</span>' for r in data.get('relevant_regulations', [])), unsafe_allow_html=True)
                        updates = data.get('updates', [])
                        if updates:
                            st.markdown("<br>**Recent Updates**")
                            for u in updates:
                                st.markdown(f'<div class="update-card"><div class="update-title">{u.get("title","")}</div><div class="update-source">{u.get("source","")}</div></div>', unsafe_allow_html=True)
        else:
            st.info("Upload and analyse a document to view compliance requirements.")

    # ═══════════════════ TAB 6 ═══════════════════
    with tab6:
        sec_header("01", "Document-Specific Legal Updates")
        if st.session_state.document_processed and hasattr(st.session_state, 'legal_updates'):
            legal_updates = st.session_state.legal_updates
            if not legal_updates:
                st.info("No relevant legal updates found.")
            else:
                for category, data in legal_updates.items():
                    with st.expander(f"{category}  ·  {data['confidence']:.2f} confidence"):
                        if not data['updates']:
                            st.info(f"No recent updates for {category}.")
                        else:
                            for u in data['updates']:
                                st.markdown(f'<div class="update-card"><div class="update-title">{u["title"]}</div><div class="update-source">{u["source"]}</div><div class="update-snippet">{u.get("snippet","")}</div></div>', unsafe_allow_html=True)
        else:
            st.info("Upload and analyse a document to fetch legal updates.")

    # ═══════════════════ TAB 7 ═══════════════════
    with tab7:
        sec_header("01", "PDF Report")
        if st.session_state.document_processed:
            cr1, cr2 = st.columns(2)
            with cr1:
                if st.button("⊞  Compile Full Report PDF", use_container_width=True):
                    with st.spinner("Compiling…"):
                        pdf_buffer = generate_pdf(st.session_state.summaries['document'], st.session_state.risk_data, getattr(st.session_state,'legal_updates',None), getattr(st.session_state,'document_compliance',None))
                        st.session_state.pdf_buffer = pdf_buffer
                        st.success("Report compiled.")
            with cr2:
                if st.session_state.get('pdf_buffer'):
                    st.download_button("↓  Download Full Report", data=st.session_state.pdf_buffer, file_name="Legal_Analysis_Report.pdf", mime="application/pdf", use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            sec_header("02", "Email Distribution")
            email = st.text_input("Recipient Email Address", placeholder="legal@company.com")
            if email and not validate_email(email):
                st.error("Please enter a valid email address.")

            st.markdown("<br>", unsafe_allow_html=True)
            ec1, ec2 = st.columns(2)
            with ec1:
                if st.button("✉  Send Summary Report", use_container_width=True):
                    if not email or not validate_email(email):
                        st.warning("Enter a valid email address first.")
                    else:
                        with st.spinner("Sending…"):
                            summary_pdf = generate_pdf(st.session_state.summaries['document'], None, None, None)
                            email_html = create_email_text(summary=st.session_state.summaries['document'])
                            success, message = send_email(email, summary_pdf, "Your Legal Document Summary Report", email_html, "document_summary.pdf")
                            st.success("Summary dispatched.") if success else st.error(message)
            with ec2:
                if st.button("✉  Send Complete Analysis", use_container_width=True):
                    if not email or not validate_email(email):
                        st.warning("Enter a valid email address first.")
                    elif not st.session_state.get('pdf_buffer'):
                        st.warning("Generate the full report first.")
                    else:
                        with st.spinner("Sending…"):
                            email_html = create_email_text(summary=st.session_state.summaries['document'], risk_assessment="Included in attached PDF")
                            success, message = send_email(email, st.session_state.pdf_buffer, "Your Complete Legal Document Analysis", email_html, "complete_legal_analysis.pdf")
                            st.success("Complete analysis dispatched.") if success else st.error(message)
        else:
            st.info("Upload and analyse a document first to generate reports.")

    # ── Footer ──
    st.markdown("""
    <div class="lex-footer">
        <div class="lex-footer-brand">LexAI</div>
        <div class="lex-footer-info">AI-Powered Legal Intelligence Framework · 2026</div>
        <div class="lex-footer-info">Precision · Integrity · Automation</div>
        <div class="lex-footer-info" style="margin-top: 1rem; opacity: 0.5;">Developed by Vansh Sharma · MCA AI&DS</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()