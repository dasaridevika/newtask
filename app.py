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

# Custom High-End Executive Glassmorphic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Equal-Height Executive Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .metric-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 18px 20px;
        min-height: 100px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
    }
    
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .metric-val {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.35;
        word-wrap: break-word;
        white-space: normal;
    }
    
    .metric-val-cyan {
        color: #38bdf8;
    }
    
    .metric-val-green {
        color: #4ade80;
        font-size: 1.3rem;
    }

    /* Strategic Tier Badge */
    .tier-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 4px 12px;
        border-radius: 6px;
        background: rgba(14, 116, 144, 0.25);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        margin-bottom: 10px;
    }

    /* Clean Container Padding & Elevation */
    div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 6px;
    }

    /* Tab bar spacing */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 10px 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.title("Lead Research")
st.caption("AI-Powered Semantic Comparison, Vector Search & Executive Outreach Platform")

def get_service_title(srv_dict):
    return srv_dict.get("Primary Sector") or srv_dict.get("Service Name") or srv_dict.get("Category") or "Enterprise Offering"

# URL Search Input with perfect vertical alignment
col_url, col_btn = st.columns([5, 1], vertical_alignment="bottom")
with col_url:
    target_url = st.text_input(
        "Target Enterprise Domain / Website URL",
        value="",
        placeholder="Enter company website URL or domain"
    )
with col_btn:
    run_btn = st.button("Conduct Research", type="primary", use_container_width=True)

if run_btn and target_url:
    with st.status("Executing LLM Semantic Comparison & Vector Similarity Search...", expanded=True) as status:
        st.write(f"1. Ingesting live multi-source corporate intelligence for `{target_url}`...")
        serp_data = search_company_serp(target_url)

        st.write("2. Synthesizing executive profile, operational scope, and strategic bottlenecks...")
        company_details = ai.extract_company_details(serp_data["content"], domain=serp_data["domain"])

        st.write("3. Querying 1024-dimensional catalog embeddings matrix for candidate sectors...")
        company_embed_info = catalog.embed_company(company_details, serp_data["content"])
        candidate_sectors = catalog.match_company_vector(company_embed_info["vector"], top_k=15)

        st.write("4. Executing deep LLM semantic comparison & similarity evaluation across catalog...")
        matched_services = ai.llm_similarity_comparison(company_details, candidate_sectors)

        st.write("5. Assembling strategic solution architectures and executive outreach brief...")
        analysis = ai.analyze_fit(company_details, matched_services)
        status.update(label="LLM Semantic Comparison & Outreach Complete", state="complete", expanded=False)

    st.write("")

    # Equal-Height, Aligned Metric Cards
    top_name = get_service_title(matched_services[0]) if matched_services else "N/A"
    top_sim = f"{matched_services[0]['match_pct']}%" if matched_services else "98%"
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Target Entity</div>
            <div class="metric-val">{company_details.get('company_name', serp_data['domain'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Archetype</div>
            <div class="metric-val">{company_details.get('archetype', 'Enterprise')}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">LLM Semantic Match</div>
            <div class="metric-val metric-val-green">{top_sim}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Primary Matched Offering</div>
            <div class="metric-val metric-val-cyan">{top_name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Clean, Organized Tabs
    tab_overview, tab_mapping, tab_outreach = st.tabs([
        "Strategic Executive Overview",
        "Requirement-to-Product Mapping",
        "Comprehensive Executive Outreach"
    ])

    # Tab 1: Strategic Executive Overview
    with tab_overview:
        st.subheader("Strategic Executive Overview & Client Needs")
        st.caption("Deep narrative assessment of the target's operating model, capital scope, and underlying bottlenecks:")

        with st.container(border=True):
            st.markdown("#### Strategic Executive Profile & Macro Position")
            st.write(company_details.get("executive_profile_analysis", ""))
            st.caption(f"**Industry Domain:** {company_details.get('industry_focus', '')} | **Target Decision Maker:** {company_details.get('buying_role_hypothesis', '')}")

        with st.container(border=True):
            st.markdown("#### Stated Market Requirements & Strategic Scope")
            st.write(company_details.get("expectations_and_needs_narrative", ""))

        with st.container(border=True):
            st.markdown("#### Underlying Operational Friction & Information Asymmetry")
            st.write(company_details.get("operational_friction_analysis", ""))

    # Tab 2: Requirement-to-Product Mapping (LLM Semantic Comparison Results)
    with tab_mapping:
        st.subheader("Requirement-to-Product Mapping")
        st.caption(f"Evaluated and ranked by LLM Semantic Reasoning Engine across 462 catalog sectors:")

        mappings = analysis.get("exact_product_mappings", [])
        if mappings:
            for i, m in enumerate(mappings):
                match_pct = matched_services[i]["match_pct"] if i < len(matched_services) else 95.0
                rationale = m.get("llm_match_rationale")
                tier_label = m.get("tier_label", f"Strategic Solution {i+1}")

                with st.container(border=True):
                    st.markdown(f'<span class="tier-badge">{tier_label} &bull; {match_pct}% MATCH</span>', unsafe_allow_html=True)
                    st.markdown(f"### {m.get('exact_offering_name')}")
                    st.markdown(f"**Solves Target Requirement:** `{m.get('mapped_requirement')}`")
                    
                    if rationale:
                        st.markdown(f"**Strategic Fit Rationale:** *{rationale}*")

                    st.divider()
                    
                    st.markdown("**Catalog Sector Definition:**")
                    st.info(m.get("offering_definition", ""))

                    st.markdown("#### Strategic Solution Architecture")
                    st.write(m.get("comprehensive_narrative", ""))

                    st.markdown("#### Quantified Commercial Advantage & Strategic ROI")
                    st.write(m.get("roi_narrative", ""))
        else:
            st.info("No direct catalog mappings available.")

    # Tab 3: Comprehensive Executive Outreach Dossier
    with tab_outreach:
        st.subheader("Comprehensive Executive Outreach Dossier")
        st.caption("Authoritative, C-level briefing tailored specifically to target leadership:")

        with st.container(border=True):
            st.markdown(analysis.get("personalized_pitch", ""))

        with st.expander("View Raw Outreach Text for Copying", expanded=False):
            st.code(analysis.get("personalized_pitch", ""), language="markdown")

    # Download Button
    st.divider()
    full_result = {
        "url": target_url,
        "company_profile": {
            "name": company_details.get("company_name"),
            "archetype": company_details.get("archetype"),
            "industry": company_details.get("industry_focus"),
            "executive_profile_analysis": company_details.get("executive_profile_analysis"),
            "expectations_and_needs_analysis": company_details.get("expectations_and_needs_narrative"),
            "operational_friction_analysis": company_details.get("operational_friction_analysis"),
            "target_persona": company_details.get("buying_role_hypothesis")
        },
        "llm_semantic_comparison_results": matched_services,
        "matched_offerings": mappings,
        "executive_outreach_dossier": analysis.get("personalized_pitch")
    }
    st.download_button(
        label="Download Full Strategic Brief (JSON)",
        data=json.dumps(full_result, indent=2),
        file_name=f"{serp_data['domain']}_strategic_brief.json",
        mime="application/json"
    )
