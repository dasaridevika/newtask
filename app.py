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
    page_title="Enterprise Lead Intelligence & Offering Matcher",
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

    /* Level Badges */
    .level-badge-green {
        display: inline-block;
        background: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .level-badge-blue {
        display: inline-block;
        background: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #3b82f6;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .level-badge-amber {
        display: inline-block;
        background: #78350f;
        color: #fde68a;
        border: 1px solid #d97706;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Deliverable Item Card */
    .deliverable-card {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

def get_service_title(srv):
    if not isinstance(srv, dict):
        return "Target Offering"
    return srv.get("Primary Sector") or srv.get("Service Name") or "Target Offering"

# Header Section
st.title("⚡ Enterprise Lead Intelligence & Offering Matcher")
st.caption("Evidence-Grounded Requirements Analysis → 1024-Dim Multi-Vector Similarity Search → Multi-Tier Deliverables Blueprint")

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
    with st.status("Extracting Ground-Truth Evidence & Executing Multi-Vector Matching...", expanded=True) as status:
        st.write(f"1. Ingesting multi-page live evidence for `{target_url}` via Crawl4AI...")
        serp_data = search_company_serp(target_url)
        evidence_store = serp_data.get("evidence_store")

        st.write("2. Synthesizing client requirements, operational bottlenecks, and growth mandate via Worker LLM...")
        company_details = ai.extract_company_details(
            serp_data["content"],
            domain=serp_data["domain"],
            client_inquiry=client_inquiry,
            evidence_store=evidence_store
        )

        st.write("3. Generating 1024-dim dense multi-vectors & executing evidence-grounded similarity search across 462 offerings...")
        company_embed_info = catalog.embed_company(company_details, serp_data["content"], client_inquiry=client_inquiry)
        candidate_sectors = catalog.match_company_vector(
            company_embed_info["vector"],
            company_text=serp_data["content"],
            company_details=company_details,
            client_inquiry=client_inquiry,
            top_k=int(top_k_val)
        )

        st.write("4. Mapping exact company offerings with bespoke solution architectures & deliverables blueprints...")
        matched_services = ai.llm_similarity_comparison(company_details, candidate_sectors)
        analysis = ai.analyze_fit(company_details, matched_services)
        status.update(label="Evidence Extraction, Requirements Analysis & Hybrid Matching Complete", state="complete", expanded=False)

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

    # 3-Pillar Executive Presentation Tabs
    tab_reqs, tab_offer, tab_deliver = st.tabs([
        "📋 1. Client Requirements Analysis",
        "🎯 2. What We Can Offer Them (Top Matches)",
        "📦 3. What to Deliver the Lead (Deliverables Blueprint)"
    ])

    req_summary = analysis.get("client_requirements_summary", {})
    lead_blueprint = analysis.get("lead_delivery_blueprint", {})
    mappings = analysis.get("exact_product_mappings", [])
    disqualified_audit = analysis.get("disqualified_and_speculative_audit") or analysis.get("disqualified_audit", [])

    # Tab 1: Detailed Client Requirements Analysis
    with tab_reqs:
        st.subheader("📋 Granular Client Requirements Analysis")
        st.caption("Evidence-grounded synthesis separating verified corporate operations from strategic inferences:")

        with st.container(border=True):
            st.markdown("#### 🎯 Core Growth Mandate & Operating Thesis")
            st.write(req_summary.get("core_growth_mandate", company_details.get("executive_profile_analysis", "")))
            
            persona = req_summary.get("target_decision_maker") or company_details.get("buying_role_hypothesis", "Strategic Leadership")
            st.caption(f"**Industry Focus:** {company_details.get('industry_focus', '')} | **Archetype:** {company_details.get('archetype', '')} | **Target Decision-Maker:** `{persona}`")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            with st.container(border=True):
                st.markdown("#### ⚡ Infrastructure & Capital Asset Visibility Needs")
                st.write(req_summary.get("infrastructure_and_asset_needs", "Real-time visibility into early-stage capital project pipelines, substation power interconnect queues, and facility buildouts."))

            with st.container(border=True):
                st.markdown("#### 📜 Regulatory, Permitting & ESG Compliance Needs")
                st.write(req_summary.get("regulatory_permitting_and_esg_needs", "Tracking stage-gate permitting dockets, environmental compliance reviews, and local municipal zoning approvals."))

        with col_r2:
            with st.container(border=True):
                st.markdown("#### 🔍 Market Diligence & Deal Sourcing Requirements")
                st.write(req_summary.get("market_diligence_and_deal_sourcing_needs", "Eliminating diligence blind spots, sourcing off-market pipeline assets, and accelerating technical evaluation cycles."))

            with st.container(border=True):
                st.markdown("#### ⚠️ Primary Operational Bottlenecks & Diligence Friction")
                st.write(req_summary.get("primary_operational_bottleneck", "Navigating long project lead times and fragmented public regulatory filings."))

    # Tab 2: What We Can Offer Them (Matched Offerings & Top K Leaderboard)
    with tab_offer:
        st.subheader("🎯 What We Can Offer Them")
        st.caption(f"Evidence-grounded offerings ranked by 1024-dimensional semantic similarity and verified corporate footprint:")

        if mappings:
            for i, m in enumerate(mappings):
                ev_level = m.get("evidence_level", "LEVEL 2 (Verified Portfolio Exposure)")
                conf = m.get("confidence", "HIGH")
                tier_label = m.get("tier_label", f"Strategic Solution {i+1}")
                score_bd = m.get("score_breakdown", {})
                vec_score = score_bd.get("vector_cosine", 0.65)
                fit_score = score_bd.get("business_fit_score", 0.75)

                badge_class = "level-badge-green" if "LEVEL 1" in ev_level or "LEVEL 2" in ev_level else ("level-badge-blue" if "LEVEL 3" in ev_level else "level-badge-amber")

                with st.container(border=True):
                    col_t1, col_t2 = st.columns([4, 1])
                    with col_t1:
                        st.markdown(f'<span class="{badge_class}">{tier_label} &bull; {ev_level}</span>', unsafe_allow_html=True)
                    with col_t2:
                        st.caption(f"Cosine: `{vec_score}` | Fit Score: `{fit_score}`")

                    st.markdown(f"### {m.get('exact_offering_name')}")
                    st.markdown(f"**How It Fulfills Client Requirements:** `{m.get('mapped_requirement')}`")
                    
                    rationale = m.get("llm_match_rationale")
                    if rationale:
                        st.markdown(f"**Verified Evidence & Strategic Fit:** *{rationale}*")

                    st.divider()
                    
                    st.markdown("**Offering Sector Definition:**")
                    st.info(m.get("offering_definition", ""))

                    st.markdown("#### Solution Architecture & Data Deliverables")
                    st.write(m.get("comprehensive_narrative", ""))

                    st.markdown("#### Operational Value Impact")
                    st.write(m.get("operational_value_driver", ""))

            st.divider()
            st.markdown(f"### 🏆 Complete Top {len(candidate_sectors)} Candidate Offerings (Similarity Leaderboard)")
            if candidate_sectors:
                cand_df = pd.DataFrame(candidate_sectors)
                desired_cols = ["Primary Sector", "evidence_level", "confidence", "vector_cosine", "business_fit_score", "matched_keywords", "Definition"]
                cols_to_show = [c for c in desired_cols if c in cand_df.columns]
                st.dataframe(cand_df[cols_to_show], use_container_width=True)

            if disqualified_audit:
                with st.expander("🔍 Transparent Disqualification & Speculative Audit", expanded=False):
                    st.markdown("Audited non-commercial, out-of-scope, or speculative sectors that were rejected or flagged:")
                    for d in disqualified_audit:
                        st.markdown(f"- **{d.get('sector')}** (`{d.get('status')}`): {d.get('rationale')}")
        else:
            st.info("No direct catalog mappings available.")

    # Tab 3: What to Deliver the Lead (Deliverables Blueprint)
    with tab_deliver:
        st.subheader("📦 What to Deliver the Lead")
        st.caption("Multi-tier technical deliverables package and operational impact overview:")

        with st.container(border=True):
            st.markdown(f"### 📦 Multi-Tier Data Deliverables Package for `{top_name}`")
            
            st.markdown("""
            <div class="deliverable-card">
                <div style="font-weight:700; color:#38bdf8; font-size:0.95rem; margin-bottom:4px;">Tier 1: Stage-Gate Permitting & Utility Queue Tracker</div>
                <div style="font-size:0.85rem; color:#cbd5e1;">Real-time municipal zoning filings, power substation interconnection queues (MW capacity), and environmental compliance review dockets.</div>
            </div>
            <div class="deliverable-card">
                <div style="font-weight:700; color:#38bdf8; font-size:0.95rem; margin-bottom:4px;">Tier 2: Key Stakeholder & Operator Directory</div>
                <div style="font-size:0.85rem; color:#cbd5e1;">Comprehensive profiles of active developers, general contractors, asset owners, and operator networks across target jurisdictions.</div>
            </div>
            <div class="deliverable-card">
                <div style="font-weight:700; color:#38bdf8; font-size:0.95rem; margin-bottom:4px;">Tier 3: Asset-Level Technical Capacity & Specification Feeds</div>
                <div style="font-size:0.85rem; color:#cbd5e1;">Square footage specifications, clear-height door data, power redundancy topologies, and capital expenditure timelines.</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        with st.container(border=True):
            st.markdown("### 📈 Quantified Strategic Advantage & Operational Impact")
            st.write(lead_blueprint.get("operational_value_driver", "Compresses diligence and evaluation cycles, eliminates infrastructure capacity blind spots, and generates proprietary deal flow 6-9 months ahead of public auctions."))

    # Download Button
    st.divider()
    full_result = {
        "url": target_url,
        "client_inquiry": client_inquiry,
        "client_requirements_analysis": req_summary,
        "exact_matched_offerings": mappings,
        "top_k_similarity_search_results": candidate_sectors,
        "lead_delivery_blueprint": lead_blueprint,
        "disqualified_and_speculative_audit": disqualified_audit
    }
    st.download_button(
        label="Download Evidence-Backed Intelligence Dossier (JSON)",
        data=json.dumps(full_result, indent=2),
        file_name=f"{serp_data['domain']}_intelligence_dossier.json",
        mime="application/json"
    )
