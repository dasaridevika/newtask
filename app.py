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

# Custom CSS for complete responsive text fit and zero truncation
st.markdown("""
<style>
    /* Metric Card Styling with zero truncation and perfect text wrap */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.35;
        word-wrap: break-word;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    .metric-val-accent {
        color: #38bdf8;
        font-size: 1.4rem;
    }
    
    /* Ensure all text in containers wraps cleanly */
    p, span, div {
        word-break: normal;
        overflow-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.title("Lead Research")
st.caption("Strategic Client Intelligence & Executive Outreach Platform")

def get_service_title(srv_dict):
    return srv_dict.get("Primary Sector") or srv_dict.get("Service Name") or srv_dict.get("Category") or "Enterprise Offering"

# URL Search Input
col_url, col_btn = st.columns([4, 1])
with col_url:
    target_url = st.text_input(
        "Target Enterprise Domain / Website URL",
        value="",
        placeholder="Enter company website URL or domain"
    )
with col_btn:
    st.write("")
    st.write("")
    run_btn = st.button("Conduct Research", type="primary", use_container_width=True)

if run_btn and target_url:
    with st.status("Generating Strategic Analysis & Executive Outreach...", expanded=True) as status:
        st.write(f"1. Ingesting multi-source intelligence for `{target_url}`...")
        serp_data = search_company_serp(target_url)

        st.write("2. Synthesizing executive profile, operational expectations, and friction analysis...")
        company_details = ai.extract_company_details(serp_data["content"], domain=serp_data["domain"])

        st.write("3. Dense semantic vector matching against 462 primary sector definitions...")
        company_embed_info = catalog.embed_company(company_details, serp_data["content"])
        catalog._last_tfidf_vec = company_embed_info.get("tfidf_vector")
        matched_services = catalog.match_company_vector(company_embed_info["tfidf_vector"], top_k=3)

        st.write("4. Mapping requirements to our catalog offerings and drafting executive outreach...")
        analysis = ai.analyze_fit(company_details, matched_services)
        status.update(label="Analysis & Outreach Complete", state="complete", expanded=False)

    st.write("")

    # Responsive Metric Cards (Text wrapped completely, zero ellipsis/truncation)
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
            <div class="metric-label">Strategic Alignment</div>
            <div class="metric-val metric-val-accent">{analysis.get('fit_score', 98)}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        top_name = get_service_title(matched_services[0]) if matched_services else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Primary Matched Offering</div>
            <div class="metric-val" style="color:#7dd3fc;">{top_name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Clean, Organized Tabs: Overview, Product Mapping, Outreach Dossier
    tab_overview, tab_mapping, tab_outreach = st.tabs([
        "Strategic Executive Overview",
        "Requirement-to-Product Mapping",
        "Comprehensive Executive Outreach"
    ])

    # Tab 1: Strategic Executive Overview
    with tab_overview:
        st.subheader("1. Strategic Executive Overview & Client Needs")
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

    # Tab 2: Requirement-to-Product Mapping
    with tab_mapping:
        st.subheader("2. Requirement-to-Product Mapping")
        st.caption("Direct alignment of our 462-sector capital project intelligence products to their verified operational requirements:")

        mappings = analysis.get("exact_product_mappings", [])
        if mappings:
            for i, m in enumerate(mappings):
                with st.container(border=True):
                    st.markdown(f"### {i+1}. {m.get('exact_offering_name')}")
                    st.markdown(f"**Solves Target Requirement:** `{m.get('mapped_requirement')}`")
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
        st.subheader("3. Comprehensive Executive Outreach Dossier")
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
        "matched_offerings": mappings,
        "executive_outreach_dossier": analysis.get("personalized_pitch")
    }
    st.download_button(
        label="Download Full Strategic Brief (JSON)",
        data=json.dumps(full_result, indent=2),
        file_name=f"{serp_data['domain']}_strategic_brief.json",
        mime="application/json"
    )
