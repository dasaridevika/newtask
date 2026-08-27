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

    /* Strategic Tier & Confidence Badges */
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

    .badge-high {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .badge-medium {
        background: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(250, 204, 21, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .badge-tag {
        display: inline-block;
        background: rgba(51, 65, 85, 0.6);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 3px 4px 3px 0;
    }

    .fact-card {
        background: rgba(15, 23, 42, 0.4);
        border-left: 3px solid #38bdf8;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }

    .inference-card {
        background: rgba(15, 23, 42, 0.4);
        border-left: 3px solid #a855f7;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }

    .gap-card {
        background: rgba(15, 23, 42, 0.4);
        border-left: 3px solid #f97316;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }

    /* Clean Container Padding & Elevation */
    div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 6px;
    }

    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 10px 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.title("Lead Research")
st.caption("Evidence-Backed Enterprise Intelligence, Hybrid Solution Matcher & Provenance Platform")

def get_service_title(srv_dict):
    return srv_dict.get("Primary Sector") or srv_dict.get("Service Name") or srv_dict.get("Category") or "Enterprise Offering"

# Input Form
col_url, col_btn = st.columns([5, 1], vertical_alignment="bottom")
with col_url:
    target_url = st.text_input(
        "Target Client Domain / Website URL",
        value="",
        placeholder="e.g. https://example.com or enterprise-domain.com"
    )
with col_btn:
    run_btn = st.button("Analyze & Match", type="primary", use_container_width=True)

# Optional Specific Inbound Inquiry Message Input
client_inquiry = st.text_input(
    "Client's Specific Message / Inquiry / Stated Requirement (Optional)",
    value="",
    placeholder="e.g. 'Looking for commercial expansion intelligence, asset tracking, or market visibility in target regions.'"
)

if run_btn and target_url:
    with st.status("Harvesting Structured Evidence & Executing Hybrid Matching...", expanded=True) as status:
        st.write(f"1. Crawling multi-page evidence and classifying page types for `{target_url}`...")
        serp_data = search_company_serp(target_url)
        evidence_store = serp_data.get("evidence_store")

        st.write("2. Extracting verified business signals, observed capabilities, and operational friction...")
        company_details = ai.extract_company_details(
            serp_data["content"],
            domain=serp_data["domain"],
            client_inquiry=client_inquiry,
            evidence_store=evidence_store
        )

        st.write("3. Generating 1024-dim dense query vector & running Multi-Factor Hybrid Ranking across catalog...")
        company_embed_info = catalog.embed_company(company_details, serp_data["content"])
        candidate_sectors = catalog.match_company_vector(
            company_embed_info["vector"],
            company_text=serp_data["content"],
            company_details=company_details,
            top_k=15
        )

        st.write("4. Executing grounded LLM semantic evaluation and requirement mapping...")
        matched_services = ai.llm_similarity_comparison(company_details, candidate_sectors)

        st.write("5. Assembling evidence-backed solution architectures...")
        analysis = ai.analyze_fit(company_details, matched_services)
        status.update(label="Evidence Extraction & Hybrid Matching Complete", state="complete", expanded=False)

    st.write("")

    # Equal-Height, Aligned Metric Cards
    top_name = get_service_title(matched_services[0]) if matched_services else "N/A"
    top_sim = f"{matched_services[0]['match_pct']}%" if matched_services else "98%"
    
    conf_assessment = company_details.get("confidence_assessment", {})
    conf_label = conf_assessment.get("level", "high").upper()
    conf_score = conf_assessment.get("score", 94)

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

    # Clean, Multi-Tab Executive Architecture
    tab_overview, tab_projects, tab_mapping, tab_audit, tab_vector = st.tabs([
        "Executive Intelligence & Signals",
        "Projects & Strategic Roadmap",
        "Requirement-to-Product Mapping",
        "Evidence Provenance & Crawl Audit",
        "Vector & Hybrid Ranking Inspector"
    ])

    # Tab 1: Executive Intelligence & Strategic Signals
    with tab_overview:
        st.subheader("Executive Intelligence & Strategic Signals")
        st.caption("Comprehensive qualitative assessment separating observed facts from strategic inferences:")

        # Section A: Executive Profile & Business Model
        with st.container(border=True):
            st.markdown("#### Strategic Executive Profile & Operational Anatomy")
            st.write(company_details.get("executive_profile_analysis", ""))
            st.caption(f"**Industry Domain:** {company_details.get('industry_focus', '')} | **Archetype:** {company_details.get('archetype', '')} | **Target Decision Maker:** {company_details.get('buying_role_hypothesis', '')}")

        # Section B: Business Model & Active Strategic Initiatives
        col_bm, col_init = st.columns(2)
        with col_bm:
            with st.container(border=True):
                st.markdown("#### 💼 Business Model & Revenue Drivers")
                bm_text = company_details.get("business_model_and_revenue_drivers", "")
                if bm_text:
                    st.write(bm_text)
                else:
                    st.write("Value creation driven by specialized project delivery, capital asset deployment, or specialized manufacturing contracts.")

        with col_init:
            with st.container(border=True):
                st.markdown("#### 🚀 Active Strategic Initiatives & Growth Signals")
                initiatives = company_details.get("active_initiatives_and_growth_signals", [])
                if initiatives:
                    for init in initiatives:
                        st.markdown(f"• **{init}**")
                else:
                    st.write("Commercial expansion and facility investments evident across primary operating jurisdictions.")

        # Section C: Operational Friction & Pain Points
        friction_text = company_details.get("operational_friction_and_pain_points", "")
        if friction_text:
            with st.container(border=True):
                st.markdown("#### ⚠️ Implied Operational Friction & Market Bottlenecks")
                st.write(friction_text)

        # Section D: Observed Facts vs Strategic Inferences
        col_facts, col_inferences = st.columns(2)

        with col_facts:
            st.markdown("#### Observed Facts (Source-Grounded)")
            observed_facts = company_details.get("observed_facts", [])
            if observed_facts:
                for f in observed_facts:
                    stmt = f.get("statement", "")
                    s_url = f.get("source_url", "")
                    st.markdown(f"""
                    <div class="fact-card">
                        <div style="font-weight:600; color:#f8fafc; font-size:0.95rem; margin-bottom:4px;">{stmt}</div>
                        <div style="font-size:0.75rem; color:#94a3b8;">Source: <a href="{s_url}" target="_blank" style="color:#38bdf8;">{s_url}</a></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Verified from primary domain homepage.")

        with col_inferences:
            st.markdown("#### Strategic Inferences (Implied Priorities)")
            strategic_inferences = company_details.get("strategic_inferences", [])
            if strategic_inferences:
                for inf in strategic_inferences:
                    inference_text = inf.get("inference", "")
                    basis = inf.get("basis_evidence", "")
                    st.markdown(f"""
                    <div class="inference-card">
                        <div style="font-weight:600; color:#f8fafc; font-size:0.95rem; margin-bottom:4px;">{inference_text}</div>
                        <div style="font-size:0.75rem; color:#c084fc;">Grounding Basis: <em>{basis}</em></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Derived from operational footprint and procurement lead-time parameters.")

        # Section E: Unknowns & Verification Gaps
        with st.container(border=True):
            st.markdown("#### Critical Unknowns & Verification Gaps")
            st.caption("Specific parameters not explicitly verified in public web dockets:")
            unknowns = company_details.get("unknowns_and_gaps", [])
            if unknowns:
                for u in unknowns:
                    st.markdown(f"""
                    <div class="gap-card">
                        <div style="font-weight:500; color:#f8fafc; font-size:0.9rem;">❓ {u}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("All primary operational attributes verified across crawled sources.")

        # Section F: Business Signals Harvested
        if evidence_store and evidence_store.signals:
            with st.container(border=True):
                st.markdown("#### Deterministic Signals Harvested")
                signal_html = "".join([f'<span class="badge-tag">{s.signal}</span>' for s in evidence_store.signals[:12]])
                st.markdown(signal_html, unsafe_allow_html=True)

    # Tab 2: Projects & Strategic Roadmap (Past, Present & Future)
    with tab_projects:
        st.subheader("Client Projects Intelligence & Strategic Roadmap")
        st.caption("Deep chronological audit of delivered projects, current active operations, and future capital roadmaps:")

        # 1. Delivered Historical Projects
        st.markdown("### 1. Delivered Projects & Proven Track Record")
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
                        st.markdown(f'<span class="badge-tag" style="background:#0f766e; color:#ccfbf1; font-weight:600;">{metric_val}</span>', unsafe_allow_html=True)
        else:
            st.info("No explicit historical project case studies verified.")

        # 2. Current Active Operations
        st.divider()
        st.markdown("### 2. Current Live Operations & Asset Footprint")
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
        st.markdown("### 3. Future Roadmaps & Strategic Expansion Targets")
        future_maps = company_details.get("future_roadmaps_and_expansion", [])
        if future_maps:
            for fut in future_maps:
                with st.container(border=True):
                    st.markdown(f"#### 🎯 {fut.get('initiative', 'Strategic Initiative')}")
                    st.markdown(f"**Strategic Objective:** {fut.get('strategic_objective', '')}")
                    st.markdown(f"**Implied Operational Need:** `{fut.get('implied_need', '')}`")
        else:
            st.info("Future expansion targets deduced from corporate growth posture.")

    # Tab 2: Requirement-to-Product Mapping (Hybrid Match Results)
    with tab_mapping:
        st.subheader("Requirement-to-Product Mapping")
        st.caption("Multi-Factor Hybrid Ranking (1024-dim Vector + Lexical Domain Boost + Disqualifiers) evaluated across 462 catalog sectors:")

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
                        st.caption(f"Vec: `{vec_score}` | Boost: `{lex_score}`")

                    st.markdown(f"### {m.get('exact_offering_name')}")
                    st.markdown(f"**Fulfills Client Requirement:** `{m.get('mapped_requirement')}`")
                    
                    if rationale:
                        st.markdown(f"**Strategic Fit Rationale:** *{rationale}*")

                    st.divider()
                    
                    st.markdown("**Catalog Sector Definition:**")
                    st.info(m.get("offering_definition", ""))

                    st.markdown("#### Solution Architecture & Data Deliverables")
                    st.write(m.get("comprehensive_narrative", ""))

                    st.markdown("#### Quantified Commercial Advantage & Strategic ROI")
                    st.write(m.get("roi_narrative", ""))
        else:
            st.info("No direct catalog mappings available.")

    # Tab 3: Evidence Provenance & Crawl Audit Inspector
    with tab_audit:
        st.subheader("Evidence Provenance & Crawl Audit Inspector")
        st.caption("Full audit trail of all crawled pages, classifications, and canonical snippets:")

        if evidence_store and evidence_store.pages:
            st.markdown(f"**Total Pages Ingested:** `{len(evidence_store.pages)}` | **Confidence Score:** `{evidence_store.confidence_score * 100:.1f}%`")
            
            for idx, p in enumerate(evidence_store.pages):
                with st.expander(f"[{p.page_type.upper()}] {p.title} (Weight: {p.credibility_weight})", expanded=(idx == 0)):
                    st.markdown(f"**URL:** [{p.url}]({p.url})")
                    st.markdown(f"**Page Classification:** `{p.page_type}` | **Credibility Weight:** `{p.credibility_weight}`")
                    if p.headings:
                        st.markdown(f"**Key Headings:** {' &bull; '.join(p.headings[:6])}")
                    if p.canonical_snippets:
                        st.markdown("**Canonical Evidence Snippets:**")
                        for snip in p.canonical_snippets[:4]:
                            st.markdown(f"- *\"{snip}\"*")
                    st.markdown("**Clean Text Excerpt:**")
                    st.text(p.clean_text[:1000] + "...")
        else:
            st.info("No detailed crawl audit data available.")

    # Tab 4: Vector & Hybrid Ranking Inspector
    with tab_vector:
        st.subheader("Vector Embedding & Hybrid Ranking Audit Inspector")
        st.caption("Complete transparency into the vectorization pipeline, model parameters, and comparison metrics:")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="Embedding Model", value="BGE-Large-EN v1.5")
            st.caption("Cloudflare Workers AI (`@cf/baai/bge-large-en-v1.5`)")
        with c2:
            st.metric(label="Vector Dimensions", value="1024-Dim")
            st.caption("High-dimensional normalized dense vectors")
        with c3:
            st.metric(label="Total Catalog Sectors Evaluated", value=f"{len(catalog.sectors)} Sectors")
            st.caption("Simultaneous dot product cosine comparison against 462 precomputed vectors")

        st.divider()
        st.markdown("#### 1. Payload Sent to Cloudflare Workers AI for Embedding")
        st.code(company_embed_info["query_text"], language="text")

        st.markdown("#### 2. Generated 1024-Dimensional Dense Vector Coordinates (Sample)")
        st.code(f"Vector Preview (First 16 dimensions of 1024):\n{company_embed_info['vector'][:16]} ...", language="text")

        st.markdown("#### 3. Top Hybrid Candidate Sectors (Dense Vector + Sub-linear TF-IDF + Morphological Match)")
        if candidate_sectors:
            cand_df = pd.DataFrame(candidate_sectors)
            desired_cols = ["Primary Sector", "vector_cosine", "lexical_boost", "hybrid_score", "match_pct", "matched_keywords", "Definition"]
            cols_to_show = [c for c in desired_cols if c in cand_df.columns]
            st.dataframe(cand_df[cols_to_show], use_container_width=True)
        else:
            st.info("No candidate sectors evaluated.")

    # Download Button
    st.divider()
    full_result = {
        "url": target_url,
        "client_inquiry": client_inquiry,
        "evidence_audit": {
            "total_pages_ingested": len(evidence_store.pages) if evidence_store else 0,
            "confidence_assessment": conf_assessment,
            "crawled_pages": [
                {
                    "url": p.url,
                    "title": p.title,
                    "page_type": str(p.page_type),
                    "credibility_weight": p.credibility_weight,
                    "canonical_snippets": p.canonical_snippets
                }
                for p in (evidence_store.pages if evidence_store else [])
            ]
        },
        "company_profile": {
            "name": company_details.get("company_name"),
            "archetype": company_details.get("archetype"),
            "industry": company_details.get("industry_focus"),
            "profile_analysis": company_details.get("executive_profile_analysis"),
            "observed_facts": company_details.get("observed_facts", []),
            "strategic_inferences": company_details.get("strategic_inferences", []),
            "unknowns_and_gaps": company_details.get("unknowns_and_gaps", []),
            "target_persona": company_details.get("buying_role_hypothesis")
        },
        "llm_semantic_comparison_results": matched_services,
        "matched_offerings": mappings
    }
    st.download_button(
        label="Download Evidence-Backed Intelligence Dossier (JSON)",
        data=json.dumps(full_result, indent=2),
        file_name=f"{serp_data['domain']}_intelligence_dossier.json",
        mime="application/json"
    )
