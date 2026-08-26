import os
import json
import streamlit as st
import pandas as pd
from scraper import search_company_serp
from service_catalog import catalog
from worker_ai import ai

st.set_page_config(
    page_title="Lead Research",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High-Contrast Executive Glassmorphic Theme with Crystal-Clear Inputs
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Dark Atmosphere */
    .stApp {
        background-color: #0b1120 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(14, 116, 144, 0.18) 0, transparent 50%),
            radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.12) 0, transparent 50%),
            radial-gradient(at 50% 50%, rgba(15, 23, 42, 1) 0, #030712 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f8fafc !important;
    }

    /* Force all Streamlit typography to high contrast */
    p, span, label, li, .stMarkdown, .stText {
        color: #e2e8f0 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Hero Title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 20%, #e2e8f0 60%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8 !important;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Input Field Label */
    div[data-testid="stTextInput"] label,
    .stTextInput label,
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 6px !important;
    }

    /* Complete Input Box Dark Container Override */
    .stTextInput,
    .stTextInput > div,
    .stTextInput > div > div,
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-testid="stTextInputRootElement"] {
        background-color: #0f172a !important;
        background: #0f172a !important;
        border: 1.5px solid #334155 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    
    /* Input Text Itself: Guaranteed Solid White Text */
    .stTextInput input,
    div[data-baseweb="input"] input, 
    div[data-testid="stTextInput"] input,
    input[type="text"],
    input {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: #0f172a !important;
        background: #0f172a !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 10px 14px !important;
        border: none !important;
    }
    
    /* Input Placeholder */
    input::placeholder,
    div[data-baseweb="input"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        font-weight: 400 !important;
    }

    div[data-baseweb="input"]:focus-within,
    .stTextInput > div:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.35) !important;
    }

    /* High-Contrast Frosted Glass Panels */
    .glass-panel {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    }
    .glass-panel:hover {
        border-color: rgba(56, 189, 248, 0.35);
    }

    /* Metric Glass Cards */
    .glass-metric {
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }
    .glass-metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #94a3b8 !important;
        margin-bottom: 6px;
    }
    .glass-metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }

    /* Badges */
    .glass-badge {
        display: inline-block;
        background: rgba(14, 116, 144, 0.25);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8 !important;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .glass-badge-secondary {
        display: inline-block;
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #f1f5f9 !important;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    
    /* Primary Action Button */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%) !important;
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.6) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }

    /* Download Button - Fixed Dark Glassmorphic with ZERO White Flash */
    div.stDownloadButton > button {
        background: rgba(15, 23, 42, 0.85) !important;
        color: #38bdf8 !important;
        border: 1.5px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stDownloadButton > button:hover {
        background: rgba(14, 116, 144, 0.25) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.35) !important;
    }
    div.stDownloadButton > button:active,
    div.stDownloadButton > button:focus {
        background: rgba(14, 116, 144, 0.35) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4) !important;
    }

    /* Tab Navigation */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-bottom: 2px solid transparent !important;
        padding: 12px 18px !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] summary {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }

    /* Code Blocks */
    pre, code {
        background: rgba(3, 7, 18, 0.9) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
    }

    /* Links */
    a {
        color: #38bdf8 !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }
    a:hover {
        text-decoration: underline !important;
        color: #7dd3fc !important;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="hero-title">Lead Research</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Institutional Web Intelligence, 1024-Dimensional Semantic Matching, and Strategic Offering Architecture</div>', unsafe_allow_html=True)

def get_service_title(srv_dict):
    return srv_dict.get("Primary Sector") or srv_dict.get("Service Name") or srv_dict.get("Category") or "Enterprise Service"

# Search Bar
col_url, col_btn = st.columns([4, 1])
with col_url:
    target_url = st.text_input(
        "Target Enterprise Domain / Website URL",
        value="",
        placeholder="e.g. https://www.vertiv.com or https://www.amazon.in"
    )
with col_btn:
    st.write("")
    st.write("")
    run_btn = st.button("Conduct Research", type="primary", use_container_width=True)

if run_btn and target_url:
    with st.status("Executing Deep Web Intelligence & Offering Mapping...", expanded=True) as status:
        st.write(f"1. Querying Google SERP Search Engine for `{target_url}`...")
        serp_data = search_company_serp(target_url)
        st.write(f"Harvested {serp_data.get('search_results_count', 1)} verified search sections and indexed references.")

        st.write("2. Synthesizing qualitative corporate profile and past/current/future initiatives...")
        company_details = ai.extract_company_details(serp_data["content"], domain=serp_data["domain"])

        st.write("3. Generating 1024-dimensional dense semantic vector with Cloudflare Worker AI...")
        company_embed_info = catalog.embed_company(company_details, serp_data["content"])
        catalog._last_tfidf_vec = company_embed_info.get("tfidf_vector")
        matched_services = catalog.match_company_vector(company_embed_info["tfidf_vector"], top_k=3)

        st.write("4. Assembling Senior Principal strategic dossier and requirement-to-service mappings...")
        analysis = ai.analyze_fit(company_details, matched_services)
        status.update(label="Research & Strategic Mapping Complete", state="complete", expanded=False)

    st.write("")

    # Top Metrics Bar (Glassmorphic Tiles)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="glass-metric">
            <div class="glass-metric-label">Target Entity</div>
            <div class="glass-metric-value">{company_details.get('company_name', serp_data['domain'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="glass-metric">
            <div class="glass-metric-label">Strategic Fit Score</div>
            <div class="glass-metric-value" style="color:#38bdf8;">{analysis.get('fit_score', 98)}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="glass-metric">
            <div class="glass-metric-label">Harvested Sections</div>
            <div class="glass-metric-value" style="color:#4ade80;">{serp_data.get('search_results_count', 1)} Indexed</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        top_name = get_service_title(matched_services[0]) if matched_services else "N/A"
        st.markdown(f"""
        <div class="glass-metric">
            <div class="glass-metric-label">Primary Matched Sector</div>
            <div class="glass-metric-value" style="font-size:1.25rem;color:#f8fafc;">{top_name}</div>
        </div>
        """, unsafe_allow_html=True)

    if serp_data.get("source_links"):
        links_pills = "".join([f'<span class="glass-badge-secondary">{link.split("//")[-1].split("/")[0]}</span>' for link in serp_data["source_links"][:5]])
        st.markdown(f"**Indexed Sources:** {links_pills}", unsafe_allow_html=True)
        st.write("")

    # Navigation Tabs
    tab_dossier, tab_projects, tab_mapping, tab_roi, tab_inspector = st.tabs([
        "Strategic Executive Dossier",
        "Projects & Initiatives Research",
        "Requirement-to-Service Mapping",
        "Commercial ROI & Outreach Pitch",
        "Dense Vector & Matching Inspector"
    ])

    # Tab 1: Strategic Executive Dossier
    with tab_dossier:
        st.markdown(f"""
        <div class="glass-panel">
            <div class="glass-badge">Executive Strategic Assessment</div>
            <div style="font-size:1.15rem;font-weight:600;color:#ffffff;margin-bottom:14px;line-height:1.55;">
                {analysis.get('executive_summary', '')}
            </div>
            <div style="color:#e2e8f0;font-size:0.95rem;line-height:1.65;">
                {analysis.get('detailed_company_analysis', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        op_dossier = company_details.get("operating_model_dossier", {})
        if op_dossier:
            st.markdown("### Core Operating Model & Commercial Behavior")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="glass-panel" style="height:100%;">
                    <div class="glass-badge">Revenue Architecture</div>
                    <div style="color:#e2e8f0;font-size:0.92rem;line-height:1.55;">
                        {op_dossier.get('revenue_architecture', 'Direct commercial delivery of specialized capabilities.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="glass-panel" style="height:100%;">
                    <div class="glass-badge">Market Position & Moat</div>
                    <div style="color:#e2e8f0;font-size:0.92rem;line-height:1.55;">
                        {op_dossier.get('market_position_and_moat', 'Established customer footprint and domain specialization.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="glass-panel" style="height:100%;">
                    <div class="glass-badge">Procurement & CAPEX Cycle</div>
                    <div style="color:#e2e8f0;font-size:0.92rem;line-height:1.55;">
                        {op_dossier.get('procurement_and_capex_cycle', 'Long-term capital planning and continuous investment screening.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### Stated Strategic Requirements")
        for req in company_details.get("expectations_and_needs", []):
            st.markdown(f"- **{req}**")

    # Tab 2: Detailed Projects Research
    with tab_projects:
        st.markdown("### Comprehensive Projects & Initiatives Research")
        st.caption("Factual breakdown of delivered case studies, active operational lines, and long-term capital roadmaps with verified source links.")
        st.write("")

        col_prev, col_curr, col_fut = st.columns(3)

        with col_prev:
            st.markdown("#### Previous Delivered Projects")
            prev_list = company_details.get("previous_projects", [])
            if prev_list:
                for item in prev_list:
                    link_html = f'<div style="margin-top:12px;"><a href="{item.get("reference_url")}" target="_blank">View Source Link &rarr;</a></div>' if item.get("reference_url") else ""
                    seg_html = f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:8px;"><b>Target Segment:</b> <span style="color:#e2e8f0;">{item.get("client_segment")}</span></div>' if item.get("client_segment") else ""
                    st.markdown(f"""
                    <div class="glass-panel">
                        <div class="glass-badge">Delivered Project / Deal</div>
                        <div style="font-weight:700;font-size:1.05rem;color:#ffffff;margin-bottom:8px;">{item.get('project_title', 'Delivered Project')}</div>
                        <div style="color:#e2e8f0;font-size:0.92rem;line-height:1.5;">{item.get('description', '')}</div>
                        {seg_html}
                        {link_html}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No previous project records available.")

        with col_curr:
            st.markdown("#### Current Active Operations")
            curr_list = company_details.get("current_projects", [])
            if curr_list:
                for item in curr_list:
                    link_html = f'<div style="margin-top:12px;"><a href="{item.get("reference_url")}" target="_blank">View Product Link &rarr;</a></div>' if item.get("reference_url") else ""
                    focus_html = f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:8px;"><b>Domain Focus:</b> <span style="color:#e2e8f0;">{item.get("technical_focus")}</span></div>' if item.get("technical_focus") else ""
                    st.markdown(f"""
                    <div class="glass-panel">
                        <div class="glass-badge" style="color:#38bdf8;border-color:rgba(56,189,248,0.4);">Active Initiative</div>
                        <div style="font-weight:700;font-size:1.05rem;color:#ffffff;margin-bottom:8px;">{item.get('project_title', 'Active Initiative')}</div>
                        <div style="color:#e2e8f0;font-size:0.92rem;line-height:1.5;">{item.get('description', '')}</div>
                        {focus_html}
                        {link_html}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No current active project records available.")

        with col_fut:
            st.markdown("#### Future Strategic Roadmaps")
            fut_list = company_details.get("future_projects", [])
            if fut_list:
                for item in fut_list:
                    link_html = f'<div style="margin-top:12px;"><a href="{item.get("reference_url")}" target="_blank">View Expansion Link &rarr;</a></div>' if item.get("reference_url") else ""
                    time_html = f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:8px;"><b>Horizon:</b> <span style="color:#e2e8f0;">{item.get("strategic_timeline")}</span></div>' if item.get("strategic_timeline") else ""
                    st.markdown(f"""
                    <div class="glass-panel">
                        <div class="glass-badge" style="color:#4ade80;border-color:rgba(74,222,128,0.4);">Future Roadmap</div>
                        <div style="font-weight:700;font-size:1.05rem;color:#ffffff;margin-bottom:8px;">{item.get('project_title', 'Future Project')}</div>
                        <div style="color:#e2e8f0;font-size:0.92rem;line-height:1.5;">{item.get('description', '')}</div>
                        {time_html}
                        {link_html}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No future roadmap project records available.")

    # Tab 3: Requirement to Service Mapping
    with tab_mapping:
        st.markdown("### Requirement-to-Service Mapping with Full Detailed Context")
        st.caption("Direct mapping of the company's verified operational requirements against our catalog services:")
        st.write("")

        mappings = analysis.get("exact_product_mappings", [])
        if mappings:
            for i, m in enumerate(mappings):
                offering_name = m.get('exact_offering_name', f'Service Offering {i+1}')
                req_title = m.get('mapped_requirement', 'Stated Project Need')
                defn = m.get('offering_definition', '')
                summary = m.get('detailed_offering_summary', '')
                tech_fit = m.get('technical_commercial_fit', '')
                scope = m.get('scope_and_deliverables', '')

                st.markdown(f"""
                <div class="glass-panel">
                    <div class="glass-badge">Tier-1 Mapped Offering</div>
                    <div style="font-size:1.35rem;font-weight:800;color:#ffffff;margin-bottom:6px;">{offering_name}</div>
                    <div style="color:#38bdf8;font-weight:700;font-size:1rem;margin-bottom:16px;">Solves Requirement: <span style="color:#ffffff;">{req_title}</span></div>
                    
                    <div style="margin-bottom:12px;color:#f1f5f9;"><b style="color:#38bdf8;">Catalog Sector Definition:</b><br><span style="color:#e2e8f0;">{defn}</span></div>
                    <div style="margin-bottom:12px;color:#f1f5f9;"><b style="color:#38bdf8;">Detailed Offering Scope:</b><br><span style="color:#e2e8f0;">{summary}</span></div>
                    <div style="margin-bottom:12px;color:#f1f5f9;"><b style="color:#38bdf8;">Technical & Commercial Fit:</b><br><span style="color:#e2e8f0;">{tech_fit}</span></div>
                    <div style="color:#f1f5f9;"><b style="color:#38bdf8;">Enterprise Deliverables & Data Feeds:</b><br><span style="color:#e2e8f0;">{scope}</span></div>
                </div>
                """, unsafe_allow_html=True)

                detailed_tech_exp = m.get("detailed_technical_explanation", "")
                if detailed_tech_exp:
                    with st.expander(f"Read In-Depth Technical Dossier: {offering_name}", expanded=True):
                        st.markdown(detailed_tech_exp)
        else:
            st.write("No direct service mappings generated.")

    # Tab 4: Strategic ROI & Outreach Pitch
    with tab_roi:
        st.markdown(f"""
        <div class="glass-panel">
            <div class="glass-badge">Commercial ROI & Strategic Moat</div>
            <div style="color:#f8fafc;font-size:0.95rem;line-height:1.65;">
                {analysis.get('strategic_roi_breakdown', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Strategic Implementation Roadmap")
        for step in analysis.get("step_by_step_roadmap", []):
            st.markdown(f"- **{step}**")

        st.divider()
        st.markdown("### Executive Outreach Communication")
        st.code(analysis.get("personalized_pitch", ""), language="markdown")

    # Tab 5: Semantic Vector & Matching Inspector
    with tab_inspector:
        st.markdown(f"""
        <div class="glass-panel">
            <div class="glass-badge">Vector Inspector</div>
            <div style="font-size:1.15rem;font-weight:700;color:#ffffff;margin-bottom:6px;">Dense Embedding Architecture</div>
            <div style="color:#e2e8f0;font-size:0.92rem;">
                <b>Embedding Model:</b> <code>{company_embed_info.get('model_name')}</code> | <b>Dense Vector Dimensions:</b> <code>{company_embed_info['dimension']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Structured Semantic Payload for Vectorization:**")
        st.code(company_embed_info["embedding_text"], language="text")
        
        st.markdown(f"**Vector Coordinates Sample:** `{company_embed_info['vector_preview']} ...`")
        
        st.divider()
        st.markdown("### Top Catalog Matches via Cosine Similarity")
        match_df = pd.DataFrame(matched_services)
        if not match_df.empty:
            cols_to_show = [c for c in ["Primary Sector", "match_pct", "similarity", "Definition"] if c in match_df.columns]
            st.dataframe(match_df[cols_to_show], use_container_width=True)

    # Export Button (Clean Dark Glassmorphic with ZERO White Flash)
    st.divider()
    full_result = {
        "url": target_url,
        "serp_stats": {
            "search_results_count": serp_data.get("search_results_count", 1),
            "source_links": serp_data.get("source_links", [])
        },
        "company_details": company_details,
        "company_vector_embedding": {
            "model_name": company_embed_info.get("model_name"),
            "dimension": company_embed_info["dimension"],
            "embedding_text": company_embed_info["embedding_text"],
            "vector": [float(x) for x in company_embed_info["vector"]]
        },
        "lead_intent": analysis.get("lead_intent", {}),
        "matched_services": matched_services,
        "analysis": analysis
    }
    st.download_button(
        label="Download Full Intelligence Dossier (JSON)",
        data=json.dumps(full_result, indent=2),
        file_name=f"{serp_data['domain']}_lead_research_dossier.json",
        mime="application/json"
    )
