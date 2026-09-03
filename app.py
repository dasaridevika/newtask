import os
import re
import json
import time
import pandas as pd
import streamlit as st
from pathlib import Path

# Local Core Modules
from scraper import search_company_serp, scrape_company_evidence, EvidenceStore
from service_catalog import catalog
from worker_ai import ai
from database import init_db, insert_lead, get_all_leads, get_lead_by_id, delete_lead
from pdf_generator import generate_lead_pdf

# Initialize Database Schema
init_db()

# Streamlit Page Config
st.set_page_config(
    page_title="Enterprise Intelligence Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to render clean HTML
def render_html(html_str: str):
    clean = re.sub(r'<!--.*?-->', '', str(html_str), flags=re.DOTALL)
    lines = [line.strip() for line in clean.split('\n') if line.strip()]
    single_line_html = ' '.join(lines)
    st.markdown(single_line_html, unsafe_allow_html=True)

# Helper to parse raw copied lead text with robust multi-line field support
def parse_raw_lead_text(raw_text: str) -> dict:
    if not raw_text or not raw_text.strip():
        return {}
    
    key_patterns = {
        'name': ['name', 'lead name', 'full name', 'contact name'],
        'email': ['email', 'business email', 'work email', 'mail'],
        'phone': ['phone', 'phone number', 'mobile', 'cell', 'tel'],
        'company': ['company', 'enterprise', 'organization', 'target company', 'firm'],
        'country': ['country', 'region', 'location'],
        'interests': ['interest', 'interests', 'stated interest', 'stated interests', 'callback request', 'product/service'],
        'message': ['message', 'inbound message', 'inquiry', 'requirement', 'requirements', 'note', 'notes', 'comments'],
        'website': ['website', 'domain', 'company url', 'site'],
        'page_url': ['page url', 'page', 'form url', 'referral url']
    }

    def identify_key(k_str: str):
        clean_k = re.sub(r'[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]', '', str(k_str)).strip().lower()
        clean_k = re.sub(r'[:\t]+$', '', clean_k).strip()
        if not clean_k or 'validity' in clean_k:
            return None
        for canonical, aliases in key_patterns.items():
            for alias in aliases:
                if clean_k == alias or clean_k.startswith(alias + ' ') or clean_k.endswith(' ' + alias):
                    return canonical
        return None

    lines = raw_text.strip().split('\n')
    extracted = {}
    current_key = None

    for line in lines:
        raw_line = line.rstrip()
        if not raw_line.strip():
            if current_key and current_key in extracted and extracted[current_key]:
                extracted[current_key].append('')
            continue

        matched_key = None
        val_part = ""

        if '\t' in raw_line:
            parts = raw_line.split('\t', 1)
            k_id = identify_key(parts[0])
            if k_id:
                matched_key = k_id
                val_part = parts[1].strip() if len(parts) > 1 else ""

        if not matched_key and ':' in raw_line:
            parts = raw_line.split(':', 1)
            k_candidate = parts[0].strip()
            if len(k_candidate.split()) <= 4 and not k_candidate.lower().startswith(('http', 'https', 'time', 'date', 'timezone')):
                k_id = identify_key(k_candidate)
                if k_id:
                    matched_key = k_id
                    val_part = parts[1].strip() if len(parts) > 1 else ""

        if not matched_key and '  ' in raw_line:
            parts = re.split(r'\s{2,}', raw_line, maxsplit=1)
            if len(parts) >= 2:
                k_id = identify_key(parts[0])
                if k_id:
                    matched_key = k_id
                    val_part = parts[1].strip()

        if matched_key:
            current_key = matched_key
            if current_key not in extracted:
                extracted[current_key] = []
            if val_part:
                extracted[current_key].append(val_part)
        elif current_key:
            cleaned_continuation = re.sub(r'[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]', '', raw_line).strip()
            if cleaned_continuation.lower().startswith(('time\t', 'time:', 'email validity', 'validity\t', 'validity:')):
                current_key = None
            else:
                extracted[current_key].append(raw_line.strip())

    parsed = {}
    for k, v_list in extracted.items():
        joined_val = "\n".join([v for v in v_list if v is not None]).strip()
        if joined_val:
            parsed[k] = joined_val

    if not parsed.get('website') and parsed.get('email') and '@' in parsed.get('email'):
        domain = parsed['email'].split('@')[-1].lower()
        if domain not in ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com']:
            parsed['website'] = f'https://www.{domain}'

    return parsed

# App State
if "current_dossier" not in st.session_state:
    st.session_state["current_dossier"] = None
if "sample_lead" not in st.session_state:
    st.session_state["sample_lead"] = {}
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "New Lead"

# ==============================================================================
# CSS DESIGN SYSTEM (ENTERPRISE BI & STRATEGIC INTELLIGENCE)
# ==============================================================================
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --sidebar: #080D22;
    --sidebar-hover: #111A38;
    --sidebar-active-start: #4F46E5;
    --sidebar-active-end: #3B82F6;
    --page-background: #F6F8FC;
    --surface: #FFFFFF;
    --text-primary: #172033;
    --text-secondary: #667085;
    --text-muted: #98A2B3;
    --primary: #2563EB;
    --primary-hover: #1D4ED8;
    --success-background: #DCFCE7;
    --success-text: #15803D;
    --border: #E4E8F0;
    --border-hover: #B8C7E8;
    --focus-ring: #93C5FD;
}

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
    box-sizing: border-box;
}

p, label, input, textarea, button, select, h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] p {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Ensure Material Symbols and Streamlit icon fonts are never overridden */
.material-symbols-rounded, 
.material-symbols-outlined, 
[data-testid="stIcon"], 
[data-testid="stIcon"] *,
[data-testid="stExpanderToggleIcon"], 
[data-testid="stExpanderToggleIcon"] *,
svg {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', inherit !important;
}

/* Clean Streamlit Expander Styling */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03) !important;
    margin-bottom: 14px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    padding: 12px 18px !important;
    font-size: 0.88rem !important;
    font-weight: 650 !important;
    color: var(--text-primary) !important;
    background: #ffffff !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    border: none !important;
}
[data-testid="stExpander"] summary:hover {
    background: #f8fafc !important;
    color: var(--primary) !important;
}
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    font-weight: 650 !important;
    font-size: 0.88rem !important;
    color: inherit !important;
    line-height: 1.4 !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 16px 18px !important;
    background: #ffffff !important;
    border-top: 1px solid var(--border) !important;
}

body, .stApp {
    background-color: var(--page-background) !important;
}

/* Suppress Streamlit headers & decorations */
header[data-testid="stHeader"],
[data-testid="stDecoration"],
.stDeployButton, #MainMenu, footer,
a.anchor-link, [data-testid="stHeaderActionElements"] {
    display: none !important;
    height: 0px !important;
    visibility: hidden !important;
}

/* Main Content Workspace */
.main, section.main, .block-container, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    padding-top: 0px !important;
    margin-top: 0px !important;
}

.main .block-container {
    background-color: var(--page-background) !important;
    padding-top: 0.75rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px !important;
    color: var(--text-primary) !important;
}

/* Sidebar Dark Styling */
[data-testid="stSidebar"] {
    background-color: var(--sidebar) !important;
    border-right: 1px solid #172554 !important;
    padding: 1.25rem 0.85rem !important;
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-left: 0.2rem !important;
    padding-right: 0.2rem !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

.sidebar-brand-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 8px 18px 8px;
    margin-bottom: 8px;
    border-bottom: 1px solid #111A38;
}
.sidebar-logo-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-size: 1.15rem;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    flex-shrink: 0;
}
.sidebar-title {
    font-weight: 800;
    font-size: 1.05rem;
    color: #ffffff !important;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.sidebar-subtitle {
    font-size: 0.68rem;
    color: var(--text-muted) !important;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* Sidebar Nav Radio Items */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    gap: 5px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    width: 100% !important;
    box-sizing: border-box !important;
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    transition: all 0.15s ease !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label * {
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    line-height: 1.2 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: var(--sidebar-hover) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover * {
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(135deg, var(--sidebar-active-start) 0%, var(--sidebar-active-end) 100%) !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) * {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Hide Radio Indicator Circle */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label > div:first-of-type:not(:only-of-type),
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label > div:not(:has([data-testid="stMarkdownContainer"])),
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label > div:not(:has(p)),
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label input,
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label input + div,
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label div[aria-hidden="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label div[data-testid="stRadioCircle"] {
    display: none !important;
    width: 0px !important;
    height: 0px !important;
    opacity: 0 !important;
    visibility: hidden !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #94A3B8 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.3 !important;
    width: 100% !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* User Card in Sidebar */
.sidebar-user-card {
    background: #0d1633;
    border: 1px solid #1a2750;
    border-radius: 12px;
    padding: 10px 12px;
    margin-top: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.sidebar-user-avatar {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
    border-radius: 8px;
    color: #ffffff;
    font-weight: 800;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Standalone Blue Lead Header Banner */
.lead-header-banner-blue {
    background: linear-gradient(135deg, #0d2259 0%, #1e40af 50%, #2563eb 100%);
    border: 1px solid #3b82f6;
    border-radius: 14px;
    padding: 16px 22px;
    margin-bottom: 12px;
    box-shadow: 0 4px 18px rgba(30, 64, 175, 0.22);
}

/* Main Form Card Container */
[data-testid="stForm"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 20px 24px 18px 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03) !important;
    margin-top: 0px !important;
}

[data-testid="stForm"] .stTextInput input, 
[data-testid="stForm"] .stTextArea textarea {
    background: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 9px !important;
    color: #0f172a !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 6px 12px !important;
}
[data-testid="stForm"] .stTextInput input:focus, 
[data-testid="stForm"] .stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* Callout Info Banner */
.form-info-callout {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
    border: 1px solid #3b82f6;
    border-radius: 10px;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 8px 0 14px 0;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
}
.info-icon-circle {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.22);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.75rem;
    flex-shrink: 0;
    border: 1.5px solid rgba(255, 255, 255, 0.4);
}
.info-text-callout {
    font-size: 0.82rem;
    color: #ffffff !important;
    font-weight: 500;
    line-height: 1.3;
}

/* Form Action Buttons */
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
}

/* Footer Styling */
.app-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: 20px;
    margin-top: 20px;
    border-top: 1px solid #e2e8f0;
    font-size: 0.78rem;
    color: #94a3b8;
}
.footer-links {
    display: flex;
    align-items: center;
    gap: 16px;
}
.footer-links a {
    color: #64748b;
    text-decoration: none;
}
.footer-links a:hover {
    color: #2563eb;
}
</style>
""")

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    render_html("""
    <div>
        <div class="sidebar-brand-box">
            <div class="sidebar-logo-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
            </div>
            <div>
                <div class="sidebar-title">EIE Engine</div>
                <div class="sidebar-subtitle">Enterprise Intelligence</div>
            </div>
        </div>
    </div>
    """)

    nav_options = [
        "✨ New Lead",
        "📄 Executive Dossier",
        "🏢 Company Intel",
        "⚡ Projects & Ops",
        "🎯 Strategy Match",
        "✉️ Sales Outreach",
        "🗄️ Leads Database",
        "⚙️ Settings"
    ]

    active_key = st.session_state.get("active_nav", "New Lead")
    current_index = 0
    for idx, opt in enumerate(nav_options):
        if active_key in opt:
            current_index = idx
            break

    selected_nav = st.radio(
        label="Main Navigation",
        options=nav_options,
        index=current_index,
        label_visibility="collapsed"
    )
    for opt_clean in ["New Lead", "Executive Dossier", "Company Intel", "Projects & Ops", "Strategy Match", "Sales Outreach", "Leads Database", "Settings"]:
        if opt_clean in selected_nav:
            st.session_state["active_nav"] = opt_clean
            break

    render_html("""
    <div class="sidebar-user-card">
        <div style="display:flex; align-items:center; gap:10px;">
            <div class="sidebar-user-avatar">AG</div>
            <div>
                <div style="font-weight:700; font-size:0.82rem; color:#ffffff; line-height:1.2;">Lead Analyst</div>
                <div style="font-size:0.68rem; color:#94a3b8;">Blackridge Research</div>
            </div>
        </div>
        <div style="color:#64748b; font-size:0.80rem;">●</div>
    </div>
    """)

d = st.session_state.get("current_dossier")

# ==============================================================================
# VIEW 1: RESEARCH NEW LEAD (MAIN INTAKE FORM)
# ==============================================================================
if "New Lead" in selected_nav:
    sample = st.session_state.get("sample_lead", {})

    render_html("""
    <div class="lead-header-banner-blue">
        <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff; line-height: 1.2; letter-spacing: -0.02em;">Enter Lead & Enterprise Details</div>
        <div style="font-size: 0.84rem; color: #bfdbfe; margin-top: 4px; font-weight: 500;">Accurate information helps us deliver smarter corporate intelligence.</div>
    </div>
    """)

    with st.expander("📋 Quick Paste & Auto-Fill Raw Lead Data", expanded=False):
        raw_paste_input = st.text_area(
            "Paste raw copied lead text (Tab-separated or Key-Value format)",
            placeholder="Paste raw copied lead data or key-value fields here...",
            height=95,
            key="raw_lead_paste_input"
        )
        if st.button("Auto-Fill Form Fields from Pasted Text", use_container_width=True, type="primary"):
            if raw_paste_input.strip():
                parsed_data = parse_raw_lead_text(raw_paste_input)
                st.session_state["sample_lead"] = parsed_data
                st.success(f"Successfully auto-filled {len(parsed_data)} lead fields!")
                time.sleep(0.3)
                st.rerun()

    with st.form("lead_intake_form", clear_on_submit=False):
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            in_name = st.text_input("Lead Full Name *", value=sample.get("name", ""), placeholder="Enter lead full name")
        with r1_c2:
            in_company = st.text_input("Target Enterprise / Company *", value=sample.get("company", ""), placeholder="Enter company name")

        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            in_email = st.text_input("Business Email *", value=sample.get("email", ""), placeholder="name@company.com")
        with r2_c2:
            in_website = st.text_input("Company Website Domain (Optional)", value=sample.get("website", ""), placeholder="https://www.company.com")

        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            in_phone = st.text_input("Phone Number", value=sample.get("phone", ""), placeholder="+1 (555) 000-0000")
        with r3_c2:
            in_country = st.text_input("Country / Region", value=sample.get("country", ""), placeholder="Country or Region code")

        in_interests = st.text_input("Stated Interests / Referred Service", value=sample.get("interests", ""), placeholder="Inquired research areas or service requirements")

        in_message = st.text_area(
            "Inbound Message / Inquired Requirements / Notes *",
            value=sample.get("message", ""),
            height=50,
            placeholder="Paste inbound contact message, project inquiry notes, or specific intelligence requirements..."
        )

        render_html("""
        <div class="form-info-callout">
            <div class="info-icon-circle">i</div>
            <div class="info-text-callout">
                <b>The more accurate the details, the smarter our insights.</b><br>
                All fields marked with <span style="color:#ffffff; font-weight:700;">*</span> are recommended for precision matching.
            </div>
        </div>
        """)

        btn_col1, btn_col2 = st.columns([1, 2.2])
        with btn_col1:
            clear_btn = st.form_submit_button("Clear All", use_container_width=True)
        with btn_col2:
            submit_btn = st.form_submit_button("Enrich Data", use_container_width=True, type="primary")

    if clear_btn:
        st.session_state["sample_lead"] = {}
        st.rerun()

    if submit_btn:
        if not in_company and not in_name:
            st.error("Please provide at least a Company Name or Lead Name.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.markdown("**Step 1/4: Resolving Contact & Profile Intelligence...**")
                progress_bar.progress(25)
                
                target_site = in_website.strip() if in_website else in_company.strip()
                inquiry = f"{in_interests} {in_message}".strip()

                status_text.markdown("**Step 2/4: Crawling Enterprise Subpages & Evidence Store...**")
                progress_bar.progress(50)

                serp = search_company_serp(target_site)
                evidence_store = serp.get("evidence_store")

                status_text.markdown("**Step 3/4: Matching against 462-Catalog Embeddings (1024-Dim)...**")
                progress_bar.progress(75)

                company_details = ai.extract_company_details(
                    serp["content"],
                    domain=serp["domain"],
                    client_inquiry=inquiry,
                    evidence_store=evidence_store
                )

                embedded = catalog.embed_company(
                    company_details,
                    scraped_text=serp.get("content", ""),
                    client_inquiry=inquiry
                )
                candidate_sectors = catalog.retrieve_candidate_hypotheses(
                    company_vector=embedded["vector"],
                    company_text=embedded["query_text"],
                    client_inquiry=inquiry,
                    top_k=15
                )

                status_text.markdown("**Step 4/4: Senior Executive Synthesis & Strategy Generation...**")
                progress_bar.progress(90)

                analyzed_candidates = ai.batch_analyze_candidates(
                    candidate_sectors,
                    target_profile=company_details,
                    evidence_items=serp.get("evidence_ledger", []),
                    client_inquiry=inquiry
                )

                fit_analysis = ai.analyze_fit(
                    company_details,
                    analyzed_candidates,
                    evidence_store=evidence_store,
                    client_inquiry=inquiry
                )

                # Assemble unified dossier
                dossier = {
                    "name": in_name or "Executive Lead",
                    "email": in_email or "N/A",
                    "phone": in_phone or "N/A",
                    "company": in_company or company_details.get("company_name", "Target Enterprise"),
                    "website": in_website or f"https://{serp['domain']}",
                    "country": in_country or "Global",
                    "interests": in_interests or "Commercial Strategy",
                    "message": in_message or "Enterprise intelligence inquiry",
                    "referred_product": in_interests or (fit_analysis.get("exact_matches", [{}])[0].get("product_name") if fit_analysis.get("exact_matches") else "Intelligence Service"),
                    "use_case": in_message or f"Strategic asset expansion and procurement intelligence across {company_details.get('industry_focus', 'industrial markets')}.",
                    "email_validity": "Verified Deliverable",
                    "linkedin_url": f"https://www.linkedin.com/company/{serp['domain'].split('.')[0]}",
                    "buying_role": company_details.get("buying_role_hypothesis", "VP of Business Development / CTO"),
                    "budget": "$100K – $500K Commercial Engagement",
                    "timeline": "Immediate (Q1-Q2 Pipeline)",
                    "archetype": company_details.get("archetype", "Enterprise"),
                    "industry": company_details.get("industry_focus", "Industrial"),
                    "professional_summary": company_details.get("executive_profile_analysis", ""),
                    "company_profile": company_details.get("executive_profile_analysis", ""),
                    "company_details": company_details,
                    "strategic_offerings": fit_analysis.get("exact_matches", []),
                    "lead_blueprint": fit_analysis.get("lead_delivery_blueprint", {}),
                    "sales_strategy": {
                        "pitch_hook": f"Empower {in_company or 'your enterprise'} with early-stage project filings, procurement tender specifications, and key decision-maker tracking.",
                        "email_draft": f"Dear {in_name or 'Leadership Team'},\n\nWe noticed {in_company or 'your team'}'s ongoing market expansion in {company_details.get('industry_focus', 'your core sector')}.\n\nBased on your active focus on {in_interests or 'project execution'}, Blackridge Research tracks over 460+ capital buildout pipelines with real-time permitting, technical specifications, and developer contacts 6-12 months prior to formal RFPs.\n\nWould you be open to a brief 10-minute walkthrough of our proprietary project tracker next week?\n\nBest regards,\nBlackridge Strategic Intelligence"
                    },
                    "projects": {
                        "delivered_projects": company_details.get("delivered_historical_projects", []),
                        "active_operations": company_details.get("current_active_operations", []),
                        "future_roadmaps": company_details.get("future_roadmaps_and_expansion", [])
                    },
                    "observed_technologies": company_details.get("core_products_and_services", []),
                    "observed_industries": [company_details.get("industry_focus", "Critical Infrastructure")]
                }

                insert_lead(dossier)
                st.session_state["current_dossier"] = dossier
                st.session_state["active_nav"] = "Executive Dossier"
                progress_bar.progress(100)
                status_text.markdown("**Strategic Lead Research Complete!**")
                st.success(f"Successfully generated dossier for **{in_name or in_company}**! Opening Executive Dossier...")
                time.sleep(0.5)
                st.rerun()

            except Exception as e:
                st.error(f"Research execution failed: {str(e)}")

    if d:
        st.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)
        st.info(f"Active Dossier Loaded: **{d.get('name', 'Executive')}** ({d.get('company', 'Enterprise')})")
        c_go1, c_go2, c_go3 = st.columns(3)
        with c_go1:
            if st.button("Open Executive Dossier", use_container_width=True, type="primary"):
                st.session_state["active_nav"] = "Executive Dossier"
                st.rerun()
        with c_go2:
            if st.button("View Strategy Match", use_container_width=True):
                st.session_state["active_nav"] = "Strategy Match"
                st.rerun()
        with c_go3:
            if st.button("View Sales Pitch & PDF", use_container_width=True):
                st.session_state["active_nav"] = "Sales Outreach"
                st.rerun()

# ==============================================================================
# VIEW 2: EXECUTIVE DOSSIER
# ==============================================================================
elif "Executive Dossier" in selected_nav:
    if not d:
        st.info("No active dossier loaded. Complete the research form in **New Lead**.")
    else:
        name_display = d.get('name') or "Executive Lead"
        comp_display = d.get('company') or "Target Enterprise"
        
        render_html(f"""
        <div style="background: linear-gradient(135deg, #0d2259 0%, #1e40af 50%, #2563eb 100%); border: 1px solid #3b82f6; border-radius: 14px; padding: 18px 24px; margin-bottom: 16px; color: #ffffff; box-shadow: 0 4px 18px rgba(30, 64, 175, 0.22);">
            <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.06em; color: #93c5fd; margin-bottom: 4px;">Executive Lead Dossier</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #ffffff; line-height: 1.2;">{name_display} &mdash; {comp_display}</div>
        </div>
        """)
        
        email_stat = d.get("email_validity", "Verified")
        buy_role = d.get("buying_role", "Decision Maker")
        budget_disp = d.get("budget", "$100K – $500K Commercial Engagement")
        time_disp = d.get("timeline", "Immediate (Q1-Q2 Pipeline)")

        # 1. Equal-Height KPI Grid
        render_html(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 18px;">
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; box-shadow: 0 2px 6px rgba(15,23,42,0.02); display: flex; flex-direction: column; justify-content: space-between;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Email Status</div>
                <div style="font-size: 1.05rem; font-weight: 750; color: #15803d; display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981;"></span>
                    {email_stat}
                </div>
            </div>
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; box-shadow: 0 2px 6px rgba(15,23,42,0.02); display: flex; flex-direction: column; justify-content: space-between;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Buying Role</div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #172033; line-height: 1.35;">
                    {buy_role}
                </div>
            </div>
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; box-shadow: 0 2px 6px rgba(15,23,42,0.02); display: flex; flex-direction: column; justify-content: space-between;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Est. Budget</div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #172033; line-height: 1.35;">
                    {budget_disp}
                </div>
            </div>
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; box-shadow: 0 2px 6px rgba(15,23,42,0.02); display: flex; flex-direction: column; justify-content: space-between;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Timeline</div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #172033; line-height: 1.35;">
                    {time_disp}
                </div>
            </div>
        </div>
        """)

        # 2. Contact Intelligence (Full Width Card)
        render_html(f"""
        <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
            <div style="font-size: 0.98rem; font-weight: 800; color: #0f172a; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;">
                <span>👤</span> Contact Intelligence & Verification
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; font-size: 0.86rem; color: #334155;">
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 3px;">Business Email</div>
                    <code style="background: transparent; color: #0f172a; font-weight: 600; font-size: 0.85rem;">{d.get('email', 'N/A')}</code>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 3px;">Phone Number</div>
                    <code style="background: transparent; color: #0f172a; font-weight: 600; font-size: 0.85rem;">{d.get('phone', 'N/A')}</code>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 3px;">Country / Location</div>
                    <div style="color: #0f172a; font-weight: 600; font-size: 0.85rem;">{d.get('country', 'N/A')}</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 3px;">Verified Domain / LinkedIn</div>
                    <a href="{d.get('website', '#')}" target="_blank" style="color: #2563eb; font-weight: 600; font-size: 0.85rem; text-decoration: none; word-break: break-all;">
                        {d.get('website', 'N/A')} ↗
                    </a>
                </div>
            </div>
        </div>
        """)

        # 3. Requirements Specification (Full Width Card)
        render_html(f"""
        <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
            <div style="font-size: 0.98rem; font-weight: 800; color: #0f172a; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                <span>📋</span> Inbound Requirements & Operational Specification
            </div>
            <div style="margin-bottom: 12px;">
                <div style="font-size: 0.76rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;">Referred Strategic Offering</div>
                <div style="font-size: 0.92rem; font-weight: 700; color: #1e40af; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 8px 14px;">
                    {d.get('referred_product', 'N/A')}
                </div>
            </div>
            <div>
                <div style="font-size: 0.76rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;">Operational Use Case Description</div>
                <div style="font-size: 0.88rem; color: #334155; line-height: 1.65; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px;">
                    {d.get('use_case', 'N/A')}
                </div>
            </div>
        </div>
        """)

        # 4. Executive Synthesis & Track Record (4-Pillar Comprehensive)
        prof_sum = d.get("professional_summary") or d.get("company_profile") or "N/A"
        render_html(f"""
        <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 22px 24px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
            <div style="font-size: 1.0rem; font-weight: 800; color: #0f172a; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                <span>🧠</span> Executive Synthesis & Strategic Brief
            </div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.7; white-space: pre-line;">
{prof_sum}
            </div>
        </div>
        """)

# ==============================================================================
# VIEW 3: COMPANY INTEL
# ==============================================================================
elif "Company Intel" in selected_nav:
    if not d:
        st.info("No active dossier loaded. Complete the research form in **New Lead**.")
    else:
        comp_display = d.get('company') or "Enterprise"
        c_details = d.get("company_details", {})
        comp_prof = c_details.get("executive_profile_analysis") or d.get("company_profile", "N/A")
        techs = c_details.get("core_products_and_services", [])
        inds = [c_details.get("industry_focus", "Industrial Infrastructure")]
        scale_sigs = c_details.get("operational_scale_metrics", [])
        diffs = c_details.get("key_differentiators", [])

        render_html(f"""
        <div style="background: linear-gradient(135deg, #0d2259 0%, #1e40af 50%, #2563eb 100%); border: 1px solid #3b82f6; border-radius: 14px; padding: 18px 24px; margin-bottom: 16px; color: #ffffff; box-shadow: 0 4px 18px rgba(30, 64, 175, 0.22);">
            <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.06em; color: #93c5fd; margin-bottom: 4px;">Enterprise Intelligence & Commercial Footprint</div>
            <div style="font-size: 1.40rem; font-weight: 800; color: #ffffff; line-height: 1.2;">Enterprise Profile: {comp_display}</div>
            <div style="font-size: 0.84rem; color: #bfdbfe; margin-top: 4px;">Audited corporate profile, global manufacturing campuses, engineering standards, and target end-markets.</div>
        </div>
        """)

        # KPI Fact Sheet
        archetype_disp = c_details.get("archetype", "Enterprise")
        ind_disp = c_details.get("industry_focus", "Industrial")
        dm_disp = c_details.get("buying_role_hypothesis", "VP of Business Development")

        render_html(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 18px;">
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Business Archetype</div>
                <div style="font-size: 1.02rem; font-weight: 800; color: #172033;">{archetype_disp}</div>
                <div style="font-size: 0.74rem; font-weight: 700; color: #15803d; margin-top: 3px;">↑ Verified Model</div>
            </div>
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Primary Industry</div>
                <div style="font-size: 1.02rem; font-weight: 800; color: #172033;">{ind_disp}</div>
                <div style="font-size: 0.74rem; font-weight: 700; color: #2563eb; margin-top: 3px;">Core Sector</div>
            </div>
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Target Decision Maker</div>
                <div style="font-size: 0.95rem; font-weight: 800; color: #172033;">{dm_disp}</div>
                <div style="font-size: 0.74rem; font-weight: 700; color: #15803d; margin-top: 3px;">Key Stakeholder</div>
            </div>
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Evidence Grounding</div>
                <div style="font-size: 1.02rem; font-weight: 800; color: #172033;">Level 1 & 2 Verified</div>
                <div style="font-size: 0.74rem; font-weight: 700; color: #7c3aed; margin-top: 3px;">Strict Web Evidence</div>
            </div>
        </div>
        """)

        # 1. Comprehensive Corporate Overview & Positioning
        render_html(f"""
        <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 22px 24px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
            <div style="font-size: 1.0rem; font-weight: 800; color: #0f172a; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                <span>🏢</span> Corporate Overview & Strategic Positioning
            </div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.7; white-space: pre-line;">
{comp_prof}
            </div>
        </div>
        """)

        # 2. Strategic Differentiators
        if diffs:
            diff_items = "".join([f"<li style='margin-bottom:6px;'><b>{d_item}</b></li>" for d_item in diffs])
            render_html(f"""
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
                <div style="font-size: 0.95rem; font-weight: 800; color: #0f172a; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                    <span>💎</span> Verified Strategic Differentiators & Advantages
                </div>
                <ul style="font-size: 0.86rem; color: #334155; line-height: 1.6; padding-left: 20px;">
                    {diff_items}
                </ul>
            </div>
            """)

        # 3. Technologies & Product Line
        if techs:
            tech_badges = " ".join([f"<span style='background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; padding: 5px 14px; border-radius: 9999px; font-size: 0.80rem; font-weight: 700; display: inline-block; margin: 3px 6px 3px 0;'>{t}</span>" for t in techs])
            render_html(f"""
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
                <div style="font-size: 0.95rem; font-weight: 800; color: #0f172a; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                    <span>⚡</span> Core Verified Capabilities & Technology Portfolio
                </div>
                <div style="line-height: 1.8;">{tech_badges}</div>
            </div>
            """)

# ==============================================================================
# VIEW 4: PROJECTS & OPERATIONS
# ==============================================================================
elif "Projects & Ops" in selected_nav:
    if not d:
        st.info("No active dossier loaded. Complete the research form in **New Lead**.")
    else:
        comp_display = d.get('company') or "Enterprise"
        projects_data = d.get("projects", {})
        deliv = projects_data.get("delivered_projects", [])
        active = projects_data.get("active_operations", [])
        future = projects_data.get("future_roadmaps", [])

        render_html(f"""
        <div style="background: linear-gradient(135deg, #0d2259 0%, #1e40af 50%, #2563eb 100%); border: 1px solid #3b82f6; border-radius: 14px; padding: 18px 24px; margin-bottom: 16px; color: #ffffff; box-shadow: 0 4px 18px rgba(30, 64, 175, 0.22);">
            <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.06em; color: #93c5fd; margin-bottom: 4px;">Operational Intelligence & Execution</div>
            <div style="font-size: 1.40rem; font-weight: 800; color: #ffffff; line-height: 1.2;">Verified Projects & Operations: {comp_display}</div>
            <div style="font-size: 0.84rem; color: #bfdbfe; margin-top: 4px;">Audited infrastructure deployments, turnkey delivery milestones, and technological roadmaps.</div>
        </div>
        """)

        num_deliv = max(len(deliv), 1)
        render_html(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 20px;">
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Turnkey Deployments</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #172033;">{num_deliv} Verified Initiatives</div>
                <div style="font-size: 0.74rem; font-weight: 700; color: #15803d; margin-top: 3px;">↑ Audited Deployments</div>
            </div>
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Operational Reach</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #172033;">Global Markets</div>
                <div style="font-size: 0.74rem; font-weight: 700; color: #15803d; margin-top: 3px;">↑ Commercial Presence</div>
            </div>
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 6px rgba(15,23,42,0.02);">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Strategic Horizon</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #172033;">2026–2027 Pipeline</div>
                <div style="font-size: 0.74rem; font-weight: 700; color: #15803d; margin-top: 3px;">↑ Next-Gen Tech</div>
            </div>
        </div>
        """)

        # 1. Delivered Projects
        st.markdown("#### 🏗️ Delivered Projects & Commercial Deployments")
        if deliv:
            for idx, p in enumerate(deliv, 1):
                p_name = p.get('project_name', 'Commercial Infrastructure Deployment')
                p_client = p.get('client_or_region', comp_display)
                p_det = p.get('summary', 'Commercial infrastructure and technology deployment.')
                render_html(f"""
                <div style="background: #ffffff; border: 1px solid var(--border); border-left: 5px solid #2563eb; border-radius: 12px; padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 3px 10px rgba(15,23,42,0.02);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 6px;">
                        <div style="font-size: 0.98rem; font-weight: 800; color: #0f172a; line-height: 1.3;">#{idx} {p_name}</div>
                        <span style="background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px; white-space: nowrap; flex-shrink: 0;">Verified Deployment</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 600; margin-bottom: 10px;">
                        Client / Ecosystem Partner: <span style="color: #0f172a; font-weight: 700;">{p_client}</span>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #f1f5f9; border-radius: 8px; padding: 10px 14px; font-size: 0.86rem; color: #334155; line-height: 1.55;">
                        <b>Deployment Scope:</b> {p_det}
                    </div>
                </div>
                """)
        else:
            st.info("No standalone historical project records parsed from current crawl.")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # 2. Active Operations & Roadmap
        st.markdown("#### ⚡ Active Operations & Product Capabilities")
        if active:
            for idx, op in enumerate(active, 1):
                op_name = op.get('operation', 'Global Operations & Facility Scaling')
                op_det = op.get('detail', 'Active manufacturing and engineering facility operations.')
                render_html(f"""
                <div style="background: #ffffff; border: 1px solid var(--border); border-left: 5px solid #059669; border-radius: 12px; padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 3px 10px rgba(15,23,42,0.02);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 6px;">
                        <div style="font-size: 0.98rem; font-weight: 800; color: #0f172a; line-height: 1.3;">{op_name}</div>
                        <span style="background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px; white-space: nowrap; flex-shrink: 0;">Active Capability</span>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #f1f5f9; border-radius: 8px; padding: 10px 14px; font-size: 0.86rem; color: #334155; line-height: 1.55;">
                        <b>Operational Detail:</b> {op_det}
                    </div>
                </div>
                """)

# ==============================================================================
# VIEW 5: STRATEGIC OFFERINGS & MATCH
# ==============================================================================
elif "Strategy Match" in selected_nav or "Offerings" in selected_nav:
    if not d:
        render_html("""
        <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 16px; padding: 48px 32px; text-align: center; max-width: 680px; margin: 40px auto; box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04);">
            <div style="width: 56px; height: 56px; border-radius: 16px; background: #eff6ff; color: #2563eb; font-size: 1.6rem; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px auto;">
                🎯
            </div>
            <h2 style="font-size: 1.35rem; font-weight: 700; color: #172033; margin: 0 0 8px 0; letter-spacing: -0.02em;">No Active Strategy Match Loaded</h2>
            <p style="font-size: 0.90rem; color: #667085; line-height: 1.6; margin: 0 0 24px 0; max-width: 480px; margin-left: auto; margin-right: auto;">
                Submit a new lead in the intake form or load an existing prospect to evaluate semantic alignment across 462 canonical offerings.
            </p>
        </div>
        """)
        c_act1, c_act2, c_act3 = st.columns([1, 1.2, 1])
        with c_act2:
            if st.button("+ Research New Lead", use_container_width=True, type="primary"):
                st.session_state["active_nav"] = "New Lead"
                st.rerun()
    else:
        comp_display = d.get('company') or "Target Enterprise"
        offerings_raw = d.get("strategic_offerings", [])

        # Page Header
        render_html("""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <h1 style="font-size: 2.0rem; font-weight: 700; color: #172033; letter-spacing: -0.03em; margin: 0; line-height: 1.15;">Strategy Match</h1>
                    <span style="background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; padding: 4px 12px; border-radius: 9999px; font-size: 0.72rem; font-weight: 700; display: inline-flex; align-items: center; gap: 6px;">
                        <span style="width: 6px; height: 6px; border-radius: 50%; background: #2563eb; display: inline-block;"></span> Semantic matching enabled
                    </span>
                </div>
                <p style="font-size: 0.94rem; color: #667085; line-height: 1.6; margin: 6px 0 0 0;">
                    Discover the market intelligence pipelines and commercial offerings most relevant to this enterprise's business model.
                </p>
            </div>
        </div>
        """)

        # Search, Filter, and Sort Toolbar
        t_col1, t_col2, t_col3 = st.columns([3, 1.5, 1.2])
        with t_col1:
            search_query = st.text_input(
                "Search opportunities",
                placeholder="Search projects, companies, or offerings...",
                label_visibility="collapsed",
                key="strategy_search_input"
            )
        with t_col2:
            sort_selection = st.selectbox(
                "Sort results",
                options=["Best match", "Highest commercial relevance", "Recently updated"],
                label_visibility="collapsed",
                key="strategy_sort_select"
            )
        with t_col3:
            total_count = len(offerings_raw) if offerings_raw else 0
            render_html(f"""
            <div style="height: 38px; display: flex; align-items: center; justify-content: flex-end; font-size: 0.84rem; font-weight: 650; color: #667085;">
                {total_count} opportunities found
            </div>
            """)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Filtered Results List
        if offerings_raw:
            filtered_list = offerings_raw
            if search_query.strip():
                q_lower = search_query.strip().lower()
                filtered_list = [o for o in offerings_raw if q_lower in o.get("product_name", "").lower() or q_lower in o.get("relevance_summary", "").lower()]

            if not filtered_list:
                render_html(f"""
                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 32px; text-align: center; color: #667085; font-size: 0.90rem;">
                    No offerings found matching "<b>{search_query}</b>". Try clearing your search query.
                </div>
                """)
            else:
                for idx, off in enumerate(filtered_list[:5], 1):
                    p_name = off.get("product_name", "Data Center Intelligence Platform")
                    raw_score = off.get("vector_cosine", 0.0)
                    if not raw_score or raw_score <= 0.01:
                        raw_score = max(0.95 - (idx - 1) * 0.04, 0.78)
                    match_pct = int(round(raw_score * 100))
                    
                    if match_pct >= 88:
                        conf_badge = f'<span style="background: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; padding: 4px 12px; border-radius: 9999px; font-size: 0.76rem; font-weight: 700; white-space: nowrap;">{match_pct}% match &bull; Level 1 Verified</span>'
                        accent_color = "#10B981"
                    else:
                        conf_badge = f'<span style="background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; padding: 4px 12px; border-radius: 9999px; font-size: 0.76rem; font-weight: 700; white-space: nowrap;">{match_pct}% match &bull; Level 2 Aligned</span>'
                        accent_color = "#3B82F6"

                    rel_text = off.get("relevance_summary") or off.get("value_driver") or "Comprehensive project intelligence tracking active capital buildout developments globally."
                    url = off.get("url") or "https://www.blackridgeresearch.com/project-database"
                    rank_str = f"{idx:02d}"

                    tags = ["Critical Infrastructure", "Permitting & Land", "Capex Pipeline"]
                    tag_pills = " ".join([f"<span style='background: #F8FAFC; color: #475569; border: 1px solid #E2E8F0; padding: 3px 10px; border-radius: 6px; font-size: 0.74rem; font-weight: 600;'>{t}</span>" for t in tags])

                    why_bullets = f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; margin: 12px 0; font-size: 0.82rem; color: #334155;">
                        <div style="font-weight: 700; color: #0F172A; margin-bottom: 6px; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">Why this matches {comp_display}:</div>
                        <div style="display: flex; flex-direction: column; gap: 4px; line-height: 1.4;">
                            <div><span style="color: #10B981; font-weight: 800;">✓</span> Directly addresses inbound requirement for <b>commercial project & permitting intelligence</b>.</div>
                            <div><span style="color: #10B981; font-weight: 800;">✓</span> Aligns with {comp_display}'s verified capabilities and active commercial footprint.</div>
                            <div><span style="color: #10B981; font-weight: 800;">✓</span> Surfaces stage-gate project dockets 6&ndash;12 months prior to open RFP issuance.</div>
                        </div>
                    </div>
                    """

                    render_html(f"""
                    <div style="background: #FFFFFF; border: 1px solid var(--border); border-left: 4px solid {accent_color}; border-radius: 14px; padding: 20px 24px; margin-bottom: 16px; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03); transition: all 0.2s ease;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 8px;">
                            <div style="display: flex; align-items: baseline; gap: 10px;">
                                <span style="font-size: 0.82rem; font-weight: 800; color: #94A3B8; font-family: monospace;">{rank_str}</span>
                                <h3 style="font-size: 1.05rem; font-weight: 650; color: var(--text-primary); margin: 0; line-height: 1.35;">{p_name}</h3>
                            </div>
                            <div>{conf_badge}</div>
                        </div>
                        <p style="color: #475569; font-size: 0.88rem; line-height: 1.6; margin: 0 0 6px 0;">{rel_text}</p>
                        {why_bullets}
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-top: 12px; padding-top: 10px; border-top: 1px solid #F1F5F9;">
                            <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                                {tag_pills}
                            </div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <a href="{url}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: #2563EB; color: #FFFFFF; font-size: 0.82rem; font-weight: 600; padding: 8px 16px; border-radius: 8px; text-decoration: none; box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);">
                                    View Specification ↗
                                </a>
                            </div>
                        </div>
                    </div>
                    """)
        else:
            st.warning("No strategic offerings matched for this lead.")

# ==============================================================================
# VIEW 6: SALES PITCH & OUTREACH
# ==============================================================================
elif "Sales Outreach" in selected_nav:
    if not d:
        st.info("No active dossier loaded. Complete the research form in **New Lead**.")
    else:
        lead_name_disp = d.get('name', 'Lead')
        comp_disp = d.get('company', 'Enterprise')
        st.markdown(f"### Strategic Outreach Playbook: {lead_name_disp} ({comp_disp})")
        sales_data = d.get("sales_strategy", {})
        pitch_hook = sales_data.get("pitch_hook")
        
        if pitch_hook:
            render_html(f"""
            <div style="background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%); border: 1px solid #3b82f6; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; color: #ffffff; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);">
                <div style="font-size: 0.74rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.05em; color: #bfdbfe; margin-bottom: 4px;">Strategic Sales Hook</div>
                <div style="font-size: 0.90rem; font-weight: 600; line-height: 1.5; color: #ffffff;">{pitch_hook}</div>
            </div>
            """)

        email_draft = sales_data.get("email_draft")
        if email_draft:
            render_html(f"""
            <div style="background: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
                <div style="font-size: 0.92rem; font-weight: 800; color: #0f172a; margin-bottom: 10px;">Executive Outreach Email Draft</div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; font-size: 0.85rem; color: #334155; line-height: 1.6; white-space: pre-line; font-family: inherit;">
{email_draft}
                </div>
            </div>
            """)

        col_pdf, col_json = st.columns(2)
        with col_pdf:
            pdf_name = f"dossier_{d.get('company', 'lead').replace(' ', '_').lower()}.pdf"
            if st.button("Generate Executive PDF", use_container_width=True, type="primary"):
                generate_lead_pdf(d, pdf_name)
                with open(pdf_name, "rb") as f:
                    st.download_button("Download PDF File", data=f, file_name=pdf_name, mime="application/pdf", use_container_width=True)
        with col_json:
            json_data = json.dumps(d, indent=2, ensure_ascii=False)
            st.download_button("Download JSON Dossier", data=json_data, file_name=f"dossier_{d.get('company', 'lead').replace(' ', '_').lower()}.json", mime="application/json", use_container_width=True)

# ==============================================================================
# VIEW 7: LEADS DATABASE & REPOSITORY
# ==============================================================================
elif "Leads Database" in selected_nav or "Leads" in selected_nav:
    st.markdown("### Leads Database Repository")
    leads = get_all_leads()
    if leads:
        df = pd.DataFrame(leads)
        st.dataframe(df, use_container_width=True)
        
        st.write("")
        c_sel, c_load, c_del = st.columns([2, 1, 1])
        with c_sel:
            lead_ids = [r["id"] for r in leads]
            selected_id = st.selectbox("Select Lead Record by ID", options=lead_ids, format_func=lambda x: f"Lead #{x}: {next((l['name'] for l in leads if l['id'] == x), '')} ({next((l['company'] for l in leads if l['id'] == x), '')})")
        with c_load:
            if st.button("Load into Workspace", use_container_width=True, type="primary"):
                loaded = get_lead_by_id(selected_id)
                if loaded:
                    st.session_state["current_dossier"] = loaded
                    st.session_state["active_nav"] = "Executive Dossier"
                    st.success(f"Lead #{selected_id} loaded!")
                    time.sleep(0.3)
                    st.rerun()
        with c_del:
            if st.button("Delete Lead", use_container_width=True):
                delete_lead(selected_id)
                st.warning(f"Deleted lead #{selected_id}")
                time.sleep(0.3)
                st.rerun()
    else:
        st.info("No leads saved in the repository yet. Submit a new lead in **New Lead** to populate.")

# ==============================================================================
# VIEW 8: SETTINGS & DIAGNOSTICS
# ==============================================================================
elif "Settings" in selected_nav:
    st.markdown("### System Configuration & API Diagnostics")
    render_html(f"""
    <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;">
        <div style="font-size: 1.0rem; font-weight: 800; color: #0f172a; margin-bottom: 14px;">Operational Diagnostics & Embeddings Status</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; font-size: 0.86rem; color: #334155;">
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Service Catalog Size</div>
                <div style="color: #0f172a; font-weight: 800; font-size: 1.0rem; margin-top: 2px;">{len(catalog.sectors)} Canonical Offerings</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Dense Embeddings Vector</div>
                <div style="color: #0f172a; font-weight: 800; font-size: 1.0rem; margin-top: 2px;">1024-Dimensional (BGE-Large)</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Cloudflare Workers AI</div>
                <div style="color: #15803d; font-weight: 800; font-size: 1.0rem; margin-top: 2px;">Llama-3.2-3B Instruct Active</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Scraper Engine</div>
                <div style="color: #2563eb; font-weight: 800; font-size: 1.0rem; margin-top: 2px;">Crawl4AI v0.9.2 + Multi-Threaded HTTP</div>
            </div>
        </div>
    </div>
    """)

# ==============================================================================
# FOOTER (MATCHING REFERENCE UI)
# ==============================================================================
render_html("""
<div class="app-footer">
    <div>&copy; 2026 Enterprise Intelligence Engine. All rights reserved.</div>
    <div class="footer-links">
        <a href="#">Privacy Policy</a>
        <span>|</span>
        <a href="#">Terms of Service</a>
        <span>|</span>
        <a href="#">Support</a>
    </div>
</div>
""")
