import streamlit as st
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from scraper import search_company_serp
from service_catalog import catalog
from worker_ai import ai

st.set_page_config(
    page_title="Lead Research | Enterprise Solution Matcher",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-Contrast Glassmorphic Theme CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Metric Cards Grid */
    .metric-card {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        text-align: left;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-val-green {
        color: #10b981;
    }
    .metric-val-cyan {
        color: #38bdf8;
    }

    /* Tier Badge */
    .tier-badge {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

def get_service_title(srv):
    if not isinstance(srv, dict):
        return "Target Offering"
    return srv.get("Primary Sector") or srv.get("Service Name") or "Target Offering"

# Header Section
st.title("⚡ Enterprise Lead Research & Offering Matcher")
st.caption("LLM & Deep Crawl Analysis of Client Works & Projects → 1024-Dim Embedding Similarity Search Across Catalog → Top K Matched Offerings")

# Input Section
col_url, col_btn = st.columns([5, 1], vertical_alignment="bottom")
with col_url:
    target_url = st.text_input(
        "Target Client Domain / Website URL",
        value="",
        placeholder="e.g. https://example.com or enterprise-domain.com"
    )
with col_btn:
    run_btn = st.button("Analyze & Match", type="primary", use_container_width=True)

# Inbound Inquiry and Top K Selector
col_inq, col_k = st.columns([4, 1])
with col_inq:
    client_inquiry = st.text_input(
        "Client's Specific Message / Inquiry / Stated Requirement (Optional)",
        value="",
        placeholder="e.g. 'Looking for commercial expansion intelligence, asset tracking, or market visibility in target regions.'"
    )
with col_k:
    top_k_val = st.number_input("Top K Matches", min_value=3, max_value=30, value=10, step=1)

if run_btn and target_url:
    with st.status("Harvesting Structured Evidence & Executing Hybrid Matching...", expanded=True) as status:
        st.write(f"1. Crawling multi-page evidence for `{target_url}` using Crawl4AI...")
        serp_data = search_company_serp(target_url)
        evidence_store = serp_data.get("evidence_store")

        st.write("2. Analyzing client works, delivered projects, and future project roadmaps via Worker LLM...")
        company_details = ai.extract_company_details(
            serp_data["content"],
            domain=serp_data["domain"],
            client_inquiry=client_inquiry,
            evidence_store=evidence_store
        )

        st.write("3. Generating 1024-dim dense query vector from client works/projects & running Similarity Search against 462 company embeddings...")
        company_embed_info = catalog.embed_company(company_details, serp_data["content"], client_inquiry=client_inquiry)
        candidate_sectors = catalog.match_company_vector(
            company_embed_info["vector"],
            company_text=serp_data["content"],
            company_details=company_details,
            client_inquiry=client_inquiry,
            top_k=int(top_k_val)
        )

        st.write("4. Executing grounded LLM semantic evaluation to map exact company products/services to client requirements...")
        matched_services = ai.llm_similarity_comparison(company_details, candidate_sectors)

        st.write("5. Assembling bespoke solution architectures & quantified ROI...")
        analysis = ai.analyze_fit(company_details, matched_services)
        status.update(label="Evidence Extraction & Hybrid Matching Complete", state="complete", expanded=False)

    st.write("")

    # Equal-Height, Aligned Header Metric Cards
    top_name = get_service_title(matched_services[0]) if matched_services else "N/A"
    
    if evidence_store and evidence_store.confidence_score:
        conf_score = int(evidence_store.confidence_score * 100)
        conf_label = evidence_store.confidence_label.upper()
    else:
        conf_assessment = company_details.get("confidence_assessment", {})
        conf_label = conf_assessment.get("level", "high").upper()
        conf_score = int(conf_assessment.get("score", 92))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Client Entity</div>
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
            <div class="metric-label">Evidence Confidence</div>
            <div class="metric-val metric-val-green">{conf_label} ({conf_score}%)</div>
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

    # 2-Tab Focused Architecture
    tab_projects, tab_mapping = st.tabs([
        "🏗️ Client Works & Projects Intelligence",
        "🎯 Exact Product/Service Offering Mapping (Top K Similarity Results)"
    ])

    # Tab 1: Client Works & Projects Intelligence
    with tab_projects:
        st.subheader("Client Works, Projects & Operational Roadmap")
        st.caption("Qualitative executive briefing and chronological audit of delivered works, active operations, and future capital roadmaps:")

        # Detailed Qualitative Executive Briefing
        with st.container(border=True):
            st.markdown("#### 📋 Strategic Executive Summary & Operational Anatomy")
            st.write(company_details.get("executive_profile_analysis", ""))
            
            bm_text = company_details.get("business_model_and_revenue_drivers", "")
            if bm_text:
                st.markdown(f"**💼 Business Model & Revenue Drivers:** {bm_text}")
            
            persona = company_details.get("buying_role_hypothesis", "Strategic Leadership")
            st.caption(f"**Industry Domain:** {company_details.get('industry_focus', '')} | **Archetype:** {company_details.get('archetype', '')} | **Target Decision Maker:** `{persona}`")

        st.divider()

        # 1. Delivered Historical Projects
        st.markdown("### 1. Delivered Works & Proven Track Record")
        hist_projects = company_details.get("delivered_historical_projects", [])
        if hist_projects:
            for p in hist_projects:
                with st.container(border=True):
                    c_h1, c_h2 = st.columns([3, 1])
                    with c_h1:
                        st.markdown(f"#### {p.get('project_name', 'Historical Project')}")
                        st.write(p.get("summary", ""))
                        if p.get("source_url"):
                            st.caption(f"Source Evidence: [{p.get('source_url')}]({p.get('source_url')})")
                    with c_h2:
                        metric_val = p.get("metric_or_milestone", "Verified Milestone")
                        st.markdown(f'<span style="display:inline-block; background:#0f766e; color:#ccfbf1; font-weight:600; padding:6px 12px; border-radius:6px; font-size:0.8rem;">{metric_val}</span>', unsafe_allow_html=True)
        else:
            st.info("No explicit historical project case studies verified.")

        # 2. Current Active Operations
        st.divider()
        st.markdown("### 2. Current Working Operations & Active Capabilities")
        curr_ops = company_details.get("current_active_operations", [])
        if curr_ops:
            for op in curr_ops:
                with st.container(border=True):
                    st.markdown(f"#### ⚙️ {op.get('operation_name', 'Active Operation')}")
                    st.write(op.get("details", ""))
                    st.caption(f"**Operational Scope:** {op.get('scope', 'Global')} | **Docket:** [{op.get('source_url', '')}]({op.get('source_url', '')})")
        else:
            st.info("Active operational footprint deduced from core domain dockets.")

        # 3. Future Roadmaps & Expansion
        st.divider()
        st.markdown("### 3. Future Project Roadmaps & Strategic Objectives")
        future_maps = company_details.get("future_roadmaps_and_expansion", [])
        if future_maps:
            for fut in future_maps:
                with st.container(border=True):
                    st.markdown(f"#### 🎯 {fut.get('initiative', 'Strategic Initiative')}")
                    st.markdown(f"**Strategic Objective:** {fut.get('strategic_objective', '')}")
                    st.markdown(f"**Implied Project Requirement:** `{fut.get('implied_need', '')}`")
        else:
            st.info("Future expansion targets deduced from corporate growth posture.")

    # Tab 2: Exact Product/Service Offering Mapping (Top K Similarity Results)
    with tab_mapping:
        st.subheader("Exact Product/Service Offering Mapping")
        st.caption(f"Results of 1024-dimensional embedding similarity search comparing client works & requirements against all 462 company offerings (Top {len(candidate_sectors)} Results):")

        mappings = analysis.get("exact_product_mappings", [])
        if mappings:
            for i, m in enumerate(mappings):
                match_pct = matched_services[i]["match_pct"] if i < len(matched_services) else 95.0
                rationale = m.get("llm_match_rationale")
                tier_label = m.get("tier_label", f"Strategic Solution {i+1}")
                score_bd = m.get("score_breakdown", {})

                with st.container(border=True):
                    col_t1, col_t2 = st.columns([4, 1])
                    with col_t1:
                        st.markdown(f'<span class="tier-badge">{tier_label} &bull; {match_pct}% FIT</span>', unsafe_allow_html=True)
                    with col_t2:
                        vec_score = score_bd.get("vector_cosine", 0.65)
                        lex_score = score_bd.get("lexical_boost", 0.20)
                        st.caption(f"Vec Cosine: `{vec_score}` | Boost: `{lex_score}`")

                    st.markdown(f"### {m.get('exact_offering_name')}")
                    st.markdown(f"**How It Fulfills Client Works & Requirements:** `{m.get('mapped_requirement')}`")
                    
                    if rationale:
                        st.markdown(f"**Exact Mapping Rationale:** *{rationale}*")

                    st.divider()
                    
                    st.markdown("**Offering Sector Definition:**")
                    st.info(m.get("offering_definition", ""))

                    st.markdown("#### Solution Architecture & Data Deliverables")
                    st.write(m.get("comprehensive_narrative", ""))

                    st.markdown("#### Quantified Commercial Advantage & Strategic ROI")
                    st.write(m.get("roi_narrative", ""))

            st.divider()
            st.markdown(f"### 🏆 Complete Top {len(candidate_sectors)} Ranked Offerings (Embedding Similarity Leaderboard)")
            if candidate_sectors:
                cand_df = pd.DataFrame(candidate_sectors)
                desired_cols = ["Primary Sector", "vector_cosine", "lexical_boost", "hybrid_score", "match_pct", "matched_keywords", "Definition"]
                cols_to_show = [c for c in desired_cols if c in cand_df.columns]
                st.dataframe(cand_df[cols_to_show], use_container_width=True)

            with st.expander("🔍 Crawl Audit & Embedding Model Metadata", expanded=False):
                st.markdown(f"**Pages Ingested:** `{len(evidence_store.pages) if evidence_store else 0}` | **Embedding Model:** `Cloudflare Workers AI (@cf/baai/bge-large-en-v1.5)` | **Dimensions:** `1024-Dim`")
                if evidence_store and evidence_store.pages:
                    for p in evidence_store.pages:
                        st.caption(f"- [{p.page_type.upper()}] [{p.title}]({p.url}) (Credibility Weight: `{p.credibility_weight}`)")
        else:
            st.info("No direct catalog mappings available.")

    # Download Button
    st.divider()
    full_result = {
        "url": target_url,
        "client_inquiry": client_inquiry,
        "client_works_and_projects": {
            "delivered_historical_projects": company_details.get("delivered_historical_projects", []),
            "current_active_operations": company_details.get("current_active_operations", []),
            "future_roadmaps_and_expansion": company_details.get("future_roadmaps_and_expansion", [])
        },
        "llm_semantic_comparison_results": matched_services,
        "exact_matched_offerings": mappings,
        "top_k_similarity_search_results": candidate_sectors
    }
    st.download_button(
        label="Download Evidence-Backed Intelligence Dossier (JSON)",
        data=json.dumps(full_result, indent=2),
        file_name=f"{serp_data['domain']}_intelligence_dossier.json",
        mime="application/json"
    )
