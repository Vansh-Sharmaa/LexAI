import re
import streamlit as st
from typing import Dict, List, Tuple

# ── Fast keyword-based document classification (no network calls) ──────────────

KEYWORD_PATTERNS = {
    "GDPR": r"\b(GDPR|General\s+Data\s+Protection\s+Regulation|data\s+protection|personal\s+data|data\s+subject|controller|processor)\b",
    "HIPAA": r"\b(HIPAA|Health\s+Insurance\s+Portability|PHI|covered\s+entity|protected\s+health\s+information)\b",
    "Contracts": r"\b(contract|agreement|parties|consideration|clause|termination|breach|obligations)\b",
    "Intellectual Property": r"\b(patent|trademark|copyright|intellectual\s+property|IP|invention|license|royalty)\b",
    "Employment Law": r"\b(employee|employer|employment|compensation|salary|benefit|termination|workplace|severance)\b",
    "Real Estate": r"\b(property|lease|tenant|landlord|real\s+estate|mortgage|rent|premises)\b",
    "Tax Law": r"\b(tax|taxation|deduction|income|exemption|IRS|audit|revenue)\b",
}

COMPLIANCE_REQUIREMENTS = {
    "GDPR": {
        "requirements": [
            "🛡️ Lawful basis for data processing documented",
            "📝 Clear privacy notice provided to data subjects",
            "🔒 Data minimization practices implemented",
            "⏱️ Right to erasure procedure established",
            "📤 Data portability mechanism available",
            "🕵️ Data Protection Impact Assessments conducted",
            "📞 Designated Data Protection Officer (if required)",
            "⚠️ 72-hour breach notification process in place",
        ],
        "relevant_regulations": [
            "EU General Data Protection Regulation (GDPR)",
            "ePrivacy Directive",
            "National Data Protection Laws",
            "Cross-Border Data Transfer Regulations",
        ],
    },
    "HIPAA": {
        "requirements": [
            "🏥 Patient authorization for PHI disclosure",
            "📁 Minimum Necessary Standard implemented",
            "🔐 Physical and technical safeguards for ePHI",
            "📝 Notice of Privacy Practices displayed",
            "👥 Workforce security training conducted",
            "📅 6-year documentation retention policy",
            "🚨 Breach notification protocol established",
            "📊 Business Associate Agreements in place",
        ],
        "relevant_regulations": [
            "Health Insurance Portability and Accountability Act (HIPAA)",
            "HITECH Act",
            "Omnibus Rule",
            "State Medical Privacy Laws",
        ],
    },
    "Contracts": {
        "requirements": [
            "📋 All parties properly identified and defined",
            "🔍 Scope of work/services clearly outlined",
            "💰 Payment terms and conditions specified",
            "⏱️ Performance timelines established",
            "🛑 Termination clauses included",
            "🔒 Confidentiality provisions included",
            "⚠️ Liability limitations specified",
            "⚖️ Governing law and dispute resolution",
        ],
        "relevant_regulations": [
            "Uniform Commercial Code (UCC)",
            "State Contract Laws",
            "Electronic Signatures in Global and National Commerce Act",
            "Foreign Corrupt Practices Act (if international)",
        ],
    },
    "Intellectual Property": {
        "requirements": [
            "🔍 Clear definition of IP assets in question",
            "🔐 Ownership rights explicitly stated",
            "📝 License terms and restrictions detailed",
            "🌎 Territorial limitations specified",
            "⏱️ Duration of rights clearly stated",
            "💰 Royalty or compensation structure",
            "⚠️ Infringement remedies outlined",
            "🔄 Rights to derivatives and improvements",
        ],
        "relevant_regulations": [
            "Copyright Act",
            "Patent Act",
            "Lanham Act (Trademarks)",
            "Defend Trade Secrets Act",
            "Digital Millennium Copyright Act",
        ],
    },
    "Employment Law": {
        "requirements": [
            "📝 Clear employment terms and conditions",
            "⏰ Working hours and overtime policies",
            "💰 Compensation and benefits structure",
            "🏥 Leave policies (sick, family, vacation)",
            "🔒 Non-disclosure and non-compete provisions",
            "⚠️ Termination procedures and severance",
            "🚫 Anti-discrimination policies",
            "👥 Employee classification (W2 vs 1099)",
        ],
        "relevant_regulations": [
            "Fair Labor Standards Act (FLSA)",
            "Family and Medical Leave Act (FMLA)",
            "Title VII of Civil Rights Act",
            "Americans with Disabilities Act (ADA)",
            "Age Discrimination in Employment Act",
        ],
    },
    "Real Estate": {
        "requirements": [
            "📍 Property clearly identified and described",
            "💰 Purchase price and payment terms",
            "🔍 Property inspection contingencies",
            "🏠 Disclosures of known defects",
            "📝 Title examination and insurance",
            "💼 Closing costs allocation",
            "📅 Timeline for closing",
            "⚠️ Default and remedy provisions",
        ],
        "relevant_regulations": [
            "State Property Laws",
            "Real Estate Settlement Procedures Act (RESPA)",
            "Truth in Lending Act (for financing)",
            "Fair Housing Act",
            "Local Zoning Ordinances",
        ],
    },
    "Tax Law": {
        "requirements": [
            "💵 Tax treatment of transactions specified",
            "📊 Record-keeping requirements",
            "🔄 Tax withholding and reporting obligations",
            "🌐 Cross-border tax considerations",
            "🏢 Entity classification for tax purposes",
            "💰 Tax indemnification provisions",
            "📝 Documentation for deductions/credits",
            "⚠️ Tax representation and warranties",
        ],
        "relevant_regulations": [
            "Internal Revenue Code",
            "State Tax Laws",
            "Foreign Account Tax Compliance Act (FATCA)",
            "Base Erosion and Profit Shifting (BEPS)",
            "Local Tax Regulations",
        ],
    },
}

LEGAL_UPDATE_SOURCES = {
    "GDPR": "https://gdpr-info.eu/",
    "HIPAA": "https://www.hhs.gov/hipaa/index.html",
    "Contracts": "https://www.americanbar.org/groups/business_law/",
    "Intellectual Property": "https://www.wipo.int/portal/en/",
    "Employment Law": "https://www.eeoc.gov/newsroom",
    "Real Estate": "https://www.hud.gov/press",
    "Tax Law": "https://www.irs.gov/newsroom",
}


def classify_document_type(text: str) -> List[Tuple[str, float]]:
    """Fast keyword-based document classification — no network calls."""
    results = []
    for category, pattern in KEYWORD_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        frequency = len(matches) / max(1, len(text.split()))
        results.append((category, round(frequency * 1000, 2)))
    return sorted(results, key=lambda x: x[1], reverse=True)


def fetch_updates_for_document(document_text: str) -> Dict:
    """
    Returns metadata about which legal domains the document touches.
    No live HTTP scraping — avoids blocking the app.
    """
    document_categories = classify_document_type(document_text)
    top_categories = document_categories[:2]

    updates: Dict = {}
    for category, confidence in top_categories:
        if confidence < 1.0:
            continue
        updates[category] = {
            "confidence": confidence,
            "updates": [
                {
                    "title": f"Review latest {category} updates",
                    "source": LEGAL_UPDATE_SOURCES.get(category, ""),
                    "snippet": (
                        f"For the most current {category} regulations and guidance, "
                        f"visit the official source linked above."
                    ),
                }
            ],
        }
    return updates


def fetch_document_compliance(document_text: str) -> Dict:
    """
    Returns compliance requirements based on keyword classification.
    Pure local computation — no external HTTP calls.
    """
    document_categories = classify_document_type(document_text)
    top_categories = document_categories[:3]

    compliance_data: Dict = {}
    for category, confidence in top_categories:
        if confidence < 0.5:
            continue
        base = COMPLIANCE_REQUIREMENTS.get(category, {})
        compliance_data[category] = {
            "confidence": confidence,
            "requirements": base.get("requirements", []),
            "relevant_regulations": base.get("relevant_regulations", []),
            "updates": [],
        }
    return compliance_data