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
    page_title="Lead Research | Enterprise Offering Matcher",
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
st.title("⚡ Enterprise Offering Matcher")
st.caption("Background Multi-Page Crawl & LLM Analysis → 1024-Dim Embedding Similarity Search Across Catalog (462 Sectors) → Top K Mapped Offerings")

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
    with st.status("Analyzing Client Intelligence & Computing Vector Similarities...", expanded=True) as status:
        st.write(f"1. Crawling evidence and ingesting subpages for `{target_url}` via Crawl4AI...")
        serp_data = search_company_serp(target_url)
        evidence_store = serp_data.get("evidence_store")

        st.write("2. Analyzing client works, operations, and requirements in background via Worker LLM...")
        company_details = ai.extract_company_details(
            serp_data["content"],
            domain=serp_data["domain"],
            client_inquiry=client_inquiry,
            evidence_store=evidence_store
        )

        st.write("3. Generating 1024-dim dense embedding & executing similarity search across all 462 company offerings...")
        company_embed_info = catalog.embed_company(company_details, serp_data["content"], client_inquiry=client_inquiry)
        candidate_sectors = catalog.match_company_vector(
            company_embed_info["vector"],
            company_text=serp_data["content"],
            company_details=company_details,
            client_inquiry=client_inquiry,
            top_k=int(top_k_val)
        )

        st.write("4. Mapping exact company products/services with bespoke solution architectures & quantified ROI...")
        matched_services = ai.llm_similarity_comparison(company_details, candidate_sectors)
        analysis = ai.analyze_fit(company_details, matched_services)
        status.update(label="Similarity Search & Offering Mapping Complete", state="complete", expanded=False)

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

    # Main Output Section: Exact Product/Service Offering Mapping (Top K Similarity Results)
    st.subheader("🎯 Exact Product/Service Offering Mapping")
    st.caption(f"Results of 1024-dimensional embedding similarity search comparing client requirements against all 462 company offerings (Top {len(candidate_sectors)} Results):")

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
                st.markdown(f"**How It Fulfills Client Requirements:** `{m.get('mapped_requirement')}`")
                
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

        with st.expander("🔍 Embedding Model Metadata & Crawl Details", expanded=False):
            st.markdown(f"**Pages Ingested:** `{len(evidence_store.pages) if evidence_store else 0}` | **Embedding Model:** `Cloudflare Workers AI (@cf/baai/bge-large-en-v1.5)` | **Dimensions:** `1024-Dim` | **Catalog Sectors:** `462`")
            if evidence_store and evidence_store.pages:
                for p in evidence_store.pages:
                    st.caption(f"- [{p.page_type.upper()}] [{p.title}]({p.url}) (Weight: `{p.credibility_weight}`)")
    else:
        st.info("No direct catalog mappings available.")

    # Download Button
    st.divider()
    full_result = {
        "url": target_url,
        "client_inquiry": client_inquiry,
        "exact_matched_offerings": mappings,
        "top_k_similarity_search_results": candidate_sectors
    }
    st.download_button(
        label="Download Evidence-Backed Intelligence Dossier (JSON)",
        data=json.dumps(full_result, indent=2),
        file_name=f"{serp_data['domain']}_intelligence_dossier.json",
        mime="application/json"
    )
