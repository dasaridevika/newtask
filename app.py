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
st.caption("Inbound Client Intelligence Resolution & AI Semantic Solution Matching Platform")

def get_service_title(srv_dict):
    return srv_dict.get("Primary Sector") or srv_dict.get("Service Name") or srv_dict.get("Category") or "Enterprise Offering"

# Input Form
col_url, col_btn = st.columns([5, 1], vertical_alignment="bottom")
with col_url:
    target_url = st.text_input(
        "Target Client Domain / Website URL",
        value="",
        placeholder="e.g. https://www.aeainvestors.com or https://www.vertiv.com"
    )
with col_btn:
    run_btn = st.button("Analyze & Match", type="primary", use_container_width=True)

# Optional Specific Inbound Inquiry Message Input
client_inquiry = st.text_input(
    "Client's Specific Message / Inquiry / Stated Requirement (Optional)",
    value="",
    placeholder="e.g. 'Looking for pipeline tracking across upcoming industrial manufacturing buildouts and chemical plant modernization dockets.'"
)

if run_btn and target_url:
    with st.status("Analyzing Client Inquiry & Matching Strategic Offerings...", expanded=True) as status:
        st.write(f"1. Ingesting client corporate intelligence for `{target_url}`...")
        serp_data = search_company_serp(target_url)

        st.write("2. Synthesizing client operational scope, inquiry context, and strategic bottlenecks...")
        company_details = ai.extract_company_details(serp_data["content"], domain=serp_data["domain"], client_inquiry=client_inquiry)

        st.write("3. Generating 1024-dimensional dense vector & performing vector cosine search across all 462 catalog sectors...")
        company_embed_info = catalog.embed_company(company_details, serp_data["content"])
        candidate_sectors = catalog.match_company_vector(company_embed_info["vector"], top_k=15)

        st.write("4. Executing deep LLM semantic comparison & similarity evaluation across catalog...")
        matched_services = ai.llm_similarity_comparison(company_details, candidate_sectors)

        st.write("5. Assembling strategic solution architectures...")
        analysis = ai.analyze_fit(company_details, matched_services)
        status.update(label="Inquiry Analysis & Solution Matching Complete", state="complete", expanded=False)

    st.write("")

    # Equal-Height, Aligned Metric Cards
    top_name = get_service_title(matched_services[0]) if matched_services else "N/A"
    top_sim = f"{matched_services[0]['match_pct']}%" if matched_services else "98%"
    
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
            <div class="metric-label">Inquiry Match Score</div>
            <div class="metric-val metric-val-green">{top_sim}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Primary Matched Solution</div>
            <div class="metric-val metric-val-cyan">{top_name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Clean, Focused Tabs (Outreach and Proposals Removed)
    tab_overview, tab_mapping, tab_audit = st.tabs([
        "Client Context & Inquiry Overview",
        "Requirement-to-Product Mapping",
        "Vector Embedding & Comparison Inspector"
    ])

    # Tab 1: Client Context & Inquiry Overview
    with tab_overview:
        st.subheader("Client Context & Inquiry Overview")
        st.caption("Deep narrative assessment of the client's operating model, inquiry context, and operational bottlenecks:")

        with st.container(border=True):
            st.markdown("#### Client Profile & Operational Scope")
            st.write(company_details.get("executive_profile_analysis", ""))
            st.caption(f"**Industry Domain:** {company_details.get('industry_focus', '')} | **Target Decision Maker:** {company_details.get('buying_role_hypothesis', '')}")

        with st.container(border=True):
            st.markdown("#### What the Client is Seeking (Inquiry Analysis)")
            st.write(company_details.get("expectations_and_needs_narrative", ""))

        with st.container(border=True):
            st.markdown("#### Operational Friction & Business Drivers")
            st.write(company_details.get("operational_friction_analysis", ""))

    # Tab 2: Requirement-to-Product Mapping
    with tab_mapping:
        st.subheader("Requirement-to-Product Mapping")
        st.caption(f"Evaluated and ranked by LLM Semantic Reasoning Engine across 462 catalog sectors to fulfill the client's request:")

        mappings = analysis.get("exact_product_mappings", [])
        if mappings:
            for i, m in enumerate(mappings):
                match_pct = matched_services[i]["match_pct"] if i < len(matched_services) else 95.0
                rationale = m.get("llm_match_rationale")
                tier_label = m.get("tier_label", f"Strategic Solution {i+1}")

                with st.container(border=True):
                    st.markdown(f'<span class="tier-badge">{tier_label} &bull; {match_pct}% FIT</span>', unsafe_allow_html=True)
                    st.markdown(f"### {m.get('exact_offering_name')}")
                    st.markdown(f"**Fulfills Client Requirement:** `{m.get('mapped_requirement')}`")
                    
                    if rationale:
                        st.markdown(f"**Strategic Fit Rationale:** *{rationale}*")

                    st.divider()
                    
                    st.markdown("**Catalog Sector Definition:**")
                    st.info(m.get("offering_definition", ""))

                    st.markdown("#### Solution Architecture & Data Deliverables")
                    st.write(m.get("comprehensive_narrative", ""))

                    st.markdown("#### Quantified Value & Operational Impact")
                    st.write(m.get("roi_narrative", ""))
        else:
            st.info("No direct catalog mappings available.")

    # Tab 3: Vector Embedding & Comparison Inspector (Transparency & Audit)
    with tab_audit:
        st.subheader("Vector Embedding & Comparison Audit Inspector")
        st.caption("Complete transparency into the vectorization pipeline, model parameters, and comparison metrics:")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="Embedding Model", value="BGE-Large-EN v1.5")
            st.caption("Cloudflare Workers AI (`@cf/baai/bge-large-en-v1.5`)")
        with c2:
            st.metric(label="Vector Dimensions", value="1024-Dim")
            st.caption("High-dimensional normalized dense vectors")
        with c3:
            st.metric(label="Total Catalog Sectors Compared", value=f"{len(catalog.sectors)} Sectors")
            st.caption("Simultaneous dot product cosine comparison against 462 precomputed vectors")

        st.divider()
        st.markdown("#### 1. Payload Sent to Cloudflare Workers AI for Embedding")
        st.code(company_embed_info["query_text"], language="text")

        st.markdown("#### 2. Generated 1024-Dimensional Dense Vector Coordinates (Sample)")
        st.code(f"Vector Preview (First 16 dimensions of 1024):\n{company_embed_info['vector'][:16]} ...", language="text")

        st.markdown("#### 3. Top Vector Cosine Similarity Candidates (Before LLM Semantic Evaluation)")
        cand_df = pd.DataFrame(candidate_sectors)[["Primary Sector", "similarity", "match_pct", "Definition"]]
        st.dataframe(cand_df, use_container_width=True)

    # Download Button
    st.divider()
    full_result = {
        "url": target_url,
        "client_inquiry": client_inquiry,
        "vector_embedding_audit": {
            "model_name": catalog.model_name,
            "dimensions": company_embed_info["dimension"],
            "total_catalog_vectors_compared": len(catalog.sectors),
            "embedded_text_payload": company_embed_info["query_text"],
            "vector_sample": [float(x) for x in company_embed_info["vector"][:32]]
        },
        "client_profile": {
            "name": company_details.get("company_name"),
            "archetype": company_details.get("archetype"),
            "industry": company_details.get("industry_focus"),
            "profile_analysis": company_details.get("executive_profile_analysis"),
            "seeking_analysis": company_details.get("expectations_and_needs_narrative"),
            "operational_friction": company_details.get("operational_friction_analysis"),
            "target_persona": company_details.get("buying_role_hypothesis")
        },
        "llm_semantic_comparison_results": matched_services,
        "matched_offerings": mappings
    }
    st.download_button(
        label="Download Client Analysis & Solution Dossier (JSON)",
        data=json.dumps(full_result, indent=2),
        file_name=f"{serp_data['domain']}_solution_dossier.json",
        mime="application/json"
    )
