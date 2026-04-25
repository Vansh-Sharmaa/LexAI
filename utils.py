import streamlit as st
import os
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer

# Load environment variables
load_dotenv()

# Constants
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

# Initialize session state with proper structure
def initialize_session_state():
    session_defaults = {
        'chat_history': [],
        'document_processed': False,
        'text_chunks': [],
        'faiss_index': None,
        'embeddings': None,
        'full_text': "",
        'comparison_result': None,
        'summaries': {},
        'risk_data': {
            'categories': {},
            'total_risks': 0,
            'severity_counts': {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            'total_score': 0
        },
        'pdf_buffer': None, # Add PDF Buffer
        'document_categories': [], # For document classification results
        'legal_updates': {},  # For storing legal updates related to the document
        'document_compliance': {},  # For storing document-specific compliance information
        'comparison_report': None  # For storing exportable comparison reports
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def get_resend_credentials():
    resend_api_key = os.getenv("RESEND_API_KEY")
    sender_email = os.getenv("SENDER_EMAIL", "onboarding@resend.dev")
    if not resend_api_key:
        raise ValueError("Missing RESEND_API_KEY in environment variables.")
    return resend_api_key, sender_email