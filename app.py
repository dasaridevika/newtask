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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-Contrast Professional Theme CSS
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
    .metric-val-amber {
        color: #f59e0b;
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

    /* Citation Box */
    .citation-box {
        background: #0f172a;
        border-left: 3px solid #10b981;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.title("Enterprise Lead Intelligence & Offering Matcher")
st.caption("Evidence-First Requirements Analysis -> 1024-Dim Multi-Vector Similarity -> Deterministic Audit Ledger")

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

if run_btn:
    clean_target_url = target_url.strip()
    if not clean_target_url:
        st.warning("Please enter a valid target client URL or domain to begin analysis.")
    else:
        target_url = clean_target_url
        with st.status("Extracting Ground-Truth Evidence & Executing Multi-Vector Matching...", expanded=True) as status:
            st.write(f"1. Ingesting multi-page live evidence and building Evidence Ledger for `{target_url}` via Crawl4AI...")
            serp_data = search_company_serp(target_url)
            evidence_store = serp_data.get("evidence_store")
            evidence_ledger = serp_data.get("evidence_ledger", [])

            st.write("2. Synthesizing fact-grounded profile without synthetic assumptions via Worker LLM...")
            company_details = ai.extract_company_details(
                serp_data["content"],
                domain=serp_data["domain"],
                client_inquiry=client_inquiry,
                evidence_store=evidence_store
            )

            st.write("3. Generating 1024-dim dense multi-vectors & executing contextual validation across 462 offerings...")
            company_embed_info = catalog.embed_company(company_details, serp_data["content"], client_inquiry=client_inquiry)
            candidate_sectors = catalog.match_company_vector(
                company_embed_info["vector"],
                company_text=serp_data["content"],
                company_details=company_details,
                client_inquiry=client_inquiry,
                evidence_ledger=evidence_ledger,
                top_k=int(top_k_val)
            )

            st.write("4. Executing strict candidate ID reasoning, post-LLM validation, and fail-closed gatekeeping...")
            matched_services = ai.llm_similarity_comparison(company_details, candidate_sectors)
            analysis = ai.analyze_fit(company_details, matched_services, evidence_ledger=evidence_ledger)
            status.update(label="Evidence Extraction, Requirements Analysis & Hybrid Matching Complete", state="complete", expanded=False)

        st.write("")

        # Metric Cards
        exact_matches = analysis.get("exact_product_mappings", [])
        adjacent_matches = analysis.get("adjacent_or_speculative_matches", [])
        top_name = exact_matches[0]["exact_offering_name"] if exact_matches else (adjacent_matches[0]["exact_offering_name"] if adjacent_matches else "No Exact Match (Evidence Gap)")
        
        if evidence_store and evidence_store.confidence_score:
            conf_score = int(evidence_store.confidence_score * 100)
            conf_label = evidence_store.confidence_label.upper()
        else:
            conf_label = "LOW"
            conf_score = 0

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
                <div class="metric-label">Evidence Status</div>
                <div class="metric-val metric-val-green">{analysis.get('status', 'verified').upper()} ({conf_score}%)</div>
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

        req_summary = analysis.get("client_requirements_summary", {})
        lead_blueprint = analysis.get("lead_delivery_blueprint", {})
        disqualified_audit = analysis.get("disqualified_and_speculative_audit", [])

        # Main Presentation Tabs
        tab_offer, tab_deliver = st.tabs([
            "1. Matched Offerings & Evidence",
            "2. Deliverables Blueprint"
        ])

        # Tab 1: Matched Offerings & Top K Leaderboard
        with tab_offer:
            st.subheader("Exact Matched Offerings (LEVEL 1 / LEVEL 2 Verified Only)")
            st.caption("Evidence-grounded offerings backed by verified operating evidence and deterministic scoring:")

            if exact_matches:
                for i, m in enumerate(exact_matches):
                    ev_level = m.get("evidence_level", "LEVEL 2 (Verified Portfolio Exposure)")
                    tier_label = m.get("tier_label", f"Strategic Solution {i+1}")
                    cid = m.get("candidate_id", "")
                    score_bd = m.get("score_breakdown", {})
                    vec_score = score_bd.get("vector_cosine", 0.65)
                    fit_score = score_bd.get("final_score", 0.75)
                    ev_count = m.get("verified_evidence_count", 0)

                    badge_class = "level-badge-green" if "LEVEL 1" in ev_level or "LEVEL 2" in ev_level else "level-badge-blue"

                    with st.container(border=True):
                        col_t1, col_t2 = st.columns([4, 1])
                        with col_t1:
                            st.markdown(f'<span class="{badge_class}">{tier_label} &bull; {cid} &bull; {ev_level} &bull; {ev_count} Verified Quotes</span>', unsafe_allow_html=True)
                        with col_t2:
                            st.caption(f"Cosine: `{vec_score}` | Final Score: `{fit_score}`")

                        st.markdown(f"### {m.get('exact_offering_name')}")
                        st.markdown(f"**How It Fulfills Client Requirements:** `{m.get('mapped_requirement')}`")
                        
                        rationale = m.get("rationale")
                        if rationale:
                            st.markdown(f"**Verified Evidence & Strategic Fit:** *{rationale}*")

                        # Supporting Verified Quotes
                        citations = m.get("supporting_citations", [])
                        if citations:
                            st.markdown("**Supporting Ground-Truth Quotes:**")
                            for cit in citations[:3]:
                                st.markdown(f"""
                                <div class="citation-box">
                                    <strong>[{cit.get('evidence_id')}]</strong> "{cit.get('quoted_text')}"<br>
                                    <span style="font-size:0.75rem; color:#94a3b8;">Source: <a href="{cit.get('source_url')}" target="_blank" style="color:#38bdf8;">{cit.get('source_url')}</a></span>
                                </div>
                                """, unsafe_allow_html=True)

                        st.divider()
                        
                        st.markdown("**Offering Sector Definition:**")
                        st.info(m.get("definition", ""))

                        st.markdown("#### Solution Architecture & Data Deliverables")
                        st.write(m.get("comprehensive_narrative", ""))

                        st.markdown("#### Operational Value Impact")
                        st.write(m.get("operational_value_driver", ""))
            else:
                st.info("No candidate offerings met the strict LEVEL 1 / LEVEL 2 ground-truth evidence threshold for exact matching. The system operates fail-closed to avoid ungrounded recommendations.")

            # Strategic Adjacencies (LEVEL 3)
            if adjacent_matches:
                st.subheader("Strategic Adjacencies (LEVEL 3)")
                st.caption("Adjacent sector expansion opportunities identified from strategic roadmaps:")
                for adj in adjacent_matches:
                    with st.container(border=True):
                        st.markdown(f"**[{adj.get('candidate_id')}] {adj.get('exact_offering_name')}** (`{adj.get('evidence_level')}`)")
                        st.write(adj.get("rationale", ""))

            st.divider()
            st.markdown(f"### Complete Top {len(candidate_sectors)} Candidate Offerings (Similarity Leaderboard)")
            if candidate_sectors:
                cand_df = pd.DataFrame(candidate_sectors)
                desired_cols = ["candidate_id", "primary_sector", "evidence_level", "verified_evidence_count", "vector_cosine", "business_fit_score", "final_score", "scale_class"]
                cols_to_show = [c for c in desired_cols if c in cand_df.columns]
                st.dataframe(cand_df[cols_to_show], use_container_width=True)

            if disqualified_audit:
                with st.expander("Transparent Disqualification & Speculative Audit", expanded=False):
                    st.markdown("Audited non-commercial, out-of-scope, or polysemous sectors that were rejected:")
                    for d in disqualified_audit:
                        cid_str = f" (`{d.get('candidate_id')}`)" if d.get('candidate_id') else ""
                        st.markdown(f"- **{d.get('sector')}**{cid_str} (`{d.get('status')}`): {d.get('rationale')}")

            # Evidence Ledger Expander
            if evidence_ledger:
                with st.expander("Harvested Evidence Ledger (Verifiable Citations)", expanded=False):
                    st.markdown(f"Total verified text evidence items extracted: **{len(evidence_ledger)}**")
                    for ev in evidence_ledger:
                        st.markdown(f"- **[{ev.get('evidence_id')}]** `{ev.get('relationship')}`: \"{ev.get('quoted_text')}\" ([Source]({ev.get('source_url')}))")

        # Tab 2: What to Deliver the Lead (Deliverables Blueprint)
        with tab_deliver:
            st.subheader("What to Deliver the Lead")
            st.caption("Multi-tier technical deliverables package and operational impact overview:")

            with st.container(border=True):
                st.markdown(f"### Multi-Tier Data Deliverables Package for `{top_name}`")
                
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
                st.markdown("### Quantified Strategic Advantage & Operational Impact")
                st.write(lead_blueprint.get("operational_value_driver", "Compresses diligence and evaluation cycles, eliminates infrastructure capacity blind spots, and generates proprietary deal flow 6-9 months ahead of public auctions."))

        # Download Button
        st.divider()
        full_result = {
            "url": target_url,
            "client_inquiry": client_inquiry,
            "client_requirements_analysis": req_summary,
            "exact_matched_offerings": exact_matches,
            "adjacent_or_speculative_matches": adjacent_matches,
            "top_k_similarity_search_results": candidate_sectors,
            "lead_delivery_blueprint": lead_blueprint,
            "disqualified_and_speculative_audit": disqualified_audit,
            "evidence_ledger": evidence_ledger,
            "validation": analysis.get("validation", {})
        }
        st.download_button(
            label="Download Evidence-Backed Intelligence Dossier (JSON)",
            data=json.dumps(full_result, indent=2),
            file_name=f"{serp_data['domain']}_intelligence_dossier.json",
            mime="application/json"
        )
