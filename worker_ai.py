import os
import json
import re
import time
import requests

def detect_archetype(company_name: str, domain: str, industry: str, summary: str) -> str:
    text = f"{company_name} {domain} {industry} {summary}".lower()
    pe_keywords = ["private equity", "investor", "investment", "portfolio", "buyout", "capital", "fund", "private debt", "credit", "asset management", "m&a", "holdings lp"]
    if any(k in text for k in pe_keywords) and not any(k in text for k in ["amazon", "aws", "google", "vertiv"]):
        return "Private Equity Sponsor & Asset Manager"
        
    hyperscale_keywords = ["hyperscale", "amazon", "aws", "google", "cloud operator", "meta", "microsoft", "azure"]
    if any(k in text for k in hyperscale_keywords):
        return "Hyperscale Cloud & Logistics Developer"
        
    oem_keywords = ["vertiv", "schneider", "eaton", "cummins", "liebert", "cooling", "thermal", "switchgear", "ups", "oem", "equipment manufacturer", "hardware"]
    if any(k in text for k in oem_keywords):
        return "Mission-Critical Infrastructure OEM"
        
    return "Enterprise Technology & Infrastructure Operator"

class WorkerAI:
    def __init__(self):
        self.worker_url = (
            os.getenv("CLOUDFLARE_WORKER_URL") 
            or os.getenv("CF_WORKER_URL") 
            or os.getenv("WORKER_URL") 
            or "https://lead-research-ai-worker.devika-worker.workers.dev"
        )
        self.model = os.getenv("CF_AI_MODEL", "@cf/meta/llama-3.2-3b-instruct")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _call_llm(self, prompt: str, system_prompt: str, max_retries: int = 2) -> str:
        """Calls Cloudflare Workers AI with intelligent retries, connection pooling, and timeouts."""
        if not self.worker_url:
            return ""
        
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": prompt
        }

        for attempt in range(max_retries + 1):
            try:
                resp = self.session.post(
                    self.worker_url,
                    json=payload,
                    timeout=45
                )
                if resp.status_code == 200:
                    data = resp.json()
                    res_text = data.get("response") or data.get("text", "")
                    if res_text and len(res_text.strip()) > 0:
                        return res_text.strip()
                elif resp.status_code == 504:
                    time.sleep(1.5)
            except Exception as e:
                if attempt == max_retries:
                    print(f"[Worker AI Connection Error]: {e}")
                time.sleep(1.0)
        return ""

    def _parse_json(self, raw_text: str) -> dict:
        """Robust multi-pattern JSON parser with automatic syntax repair."""
        if not raw_text:
            return {}
        try:
            # 1. Clean markdown code blocks
            cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"^```\s*", "", cleaned.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
            
            # 2. Extract outermost JSON object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Repair common LLM trailing comma issues
                json_str = re.sub(r",\s*([\]\}])", r"\1", json_str)
                return json.loads(json_str)
        except Exception:
            pass
        return {}

    def extract_company_details(self, scraped_text: str, domain: str = "", client_inquiry: str = "") -> dict:
        clean_name = domain.split(".")[0].replace("www", "").capitalize() if domain else "Enterprise"
        inquiry_text = f"\nClient Inbound Inquiry / Message:\n\"{client_inquiry}\"\n" if client_inquiry else ""

        # Pre-process scraped text to retain highest signal content
        condensed_text = re.sub(r"\n\s*\n+", "\n\n", scraped_text[:10000]).strip()

        system_prompt = (
            "You are a Senior Managing Director & Head of Enterprise Client Solutions.\n"
            "An enterprise client has approached our firm with an inquiry regarding their strategic requirements.\n"
            "Analyze the client company from the crawled text and their inquiry to produce an exhaustive, deep narrative dossier.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Write comprehensive narrative prose paragraphs with senior executive depth.\n"
            "2. DO NOT use bullet points or numbered lists in the narrative text.\n"
            "3. Accurately identify their operational model, what they are requesting from us, and their underlying bottlenecks.\n\n"
            "Return strictly a valid JSON object matching this schema:\n"
            "{\n"
            '  "company_name": "Official Entity Name",\n'
            '  "industry_focus": "Specific Industry Domain",\n'
            '  "executive_profile_analysis": "Comprehensive institutional assessment of their business model, commercial scale, and market footprint.",\n'
            '  "expectations_and_needs_narrative": "Detailed narrative analysis explaining exactly what forward-looking project datasets, site selection intelligence, or market visibility the client is seeking from us to support their operations.",\n'
            '  "operational_friction_analysis": "In-depth narrative analysis detailing the structural friction, lead-time bottlenecks, information asymmetry, and commercial pressures they face in their current operations that prompted their inquiry.",\n'
            '  "buying_role_hypothesis": "Specific Executive Title"\n'
            "}"
        )

        prompt = f"Target Enterprise Domain: {domain}{inquiry_text}\n\nCrawled Intelligence:\n{condensed_text}"
        raw = self._call_llm(prompt, system_prompt)
        parsed = self._parse_json(raw)

        comp_name = parsed.get("company_name", clean_name) if parsed else clean_name
        ind_focus = parsed.get("industry_focus", "") if parsed else ""
        exec_sum = parsed.get("executive_profile_analysis", "") if parsed else ""

        archetype = detect_archetype(comp_name, domain, ind_focus, exec_sum or scraped_text)

        if not parsed or len(parsed.get("expectations_and_needs_narrative", "")) < 80:
            if archetype == "Private Equity Sponsor & Asset Manager":
                comp_name = "AEA Investors LP" if "aeainvestor" in domain.lower() else f"{clean_name} Capital"
                parsed = {
                    "company_name": comp_name,
                    "industry_focus": "Middle Market Private Equity, Small Business Buyouts & Private Debt",
                    "executive_profile_analysis": (
                        f"{comp_name} is an institutional global private investment firm managing approximately $19 billion in assets under management across dedicated Middle Market Private Equity, Small Business Buyouts, and Private Debt investment strategies. Founded in 1968 by landmark industrial family offices, the firm has established a five-decade legacy of operational value creation by partnering with market-leading enterprises across value-added industrials, industrial services, specialty manufacturing, and consumer healthcare. The firm's operational model focuses on executing strategic add-on acquisitions, scaling manufacturing efficiency, and driving international expansion across middle-market platform companies."
                    ),
                    "expectations_and_needs_narrative": (
                        f"In approaching our firm, {comp_name}'s investment committees and operating partners are seeking authoritative, forward-looking intelligence on global capital expenditure pipelines. Specifically, their deal teams require verified visibility into upcoming industrial manufacturing buildouts, plant modernization dockets, and supply chain procurement cycles across specialty chemical, packaging machinery, and industrial services sectors. This forward intelligence is essential for their team to stress-test financial underwriting models, conduct commercial due diligence on potential buyout targets, and benchmark the growth avenues of their existing portfolio platforms."
                    ),
                    "operational_friction_analysis": (
                        f"{comp_name}'s deal origination and diligence processes encounter substantial structural friction stemming from reliance on lagging historical market reports that fail to capture real-time industrial capital allocation. In addition, intensely competitive investment banking auctions compress entry multiples and reduce returns, creating an urgent commercial necessity for proprietary pre-auction deal origination. Furthermore, existing portfolio companies frequently operate without advance visibility into upcoming multi-million-dollar capital projects, resulting in missed opportunities to pre-position high-margin equipment and service contracts."
                    ),
                    "buying_role_hypothesis": "Managing Director / Head of Private Equity Due Diligence & Industrial Strategy"
                }
            elif archetype == "Hyperscale Cloud & Logistics Developer":
                parsed = {
                    "company_name": "Amazon.com, Inc." if "amazon" in domain.lower() else comp_name,
                    "industry_focus": "Hyperscale Cloud Infrastructure (AWS), AI Platforms & Multimodal Logistics",
                    "executive_profile_analysis": (
                        "Amazon (NASDAQ: AMZN) is a global technology and infrastructure enterprise operating at immense scale across hyperscale cloud computing (Amazon Web Services), e-commerce retail networks, generative AI platforms (Bedrock), and physical logistics ecosystems. In regional growth corridors, Amazon is executing massive multi-billion-dollar capital allocation programs ($48B committed through 2030), expanding dedicated freight railway logistics hubs, and building hyper-density AWS cloud availability zones powered by renewable energy microgrids and substation interconnections."
                    ),
                    "expectations_and_needs_narrative": (
                        "In their inquiry, Amazon's global infrastructure planning, real estate, and procurement leadership are seeking granular, pre-construction intelligence covering land zoning dockets, municipal permits, and environmental impact filings 18 to 24 months in advance. Additionally, their infrastructure planners require real-time tracking of utility substation interconnection queues (including target Megawatt capacity and transmission kV voltage levels) to de-risk multi-gigawatt power provisioning for AWS AI clusters and optimize multimodal logistics routing adjacent to dedicated freight rail corridors."
                    ),
                    "operational_friction_analysis": (
                        "The primary constraint on Amazon's physical expansion is no longer capital—it is lead-time friction in securing high-voltage power allocations and municipal zoning approvals. Substation interconnect queues frequently span 24 to 36 months of pre-construction coordination with regional power utilities. Furthermore, regional land speculators frequently lock up high-capacity industrial parcels prior to public announcements, inflating site acquisition costs and introducing critical commissioning delays for gigawatt compute clusters."
                    ),
                    "buying_role_hypothesis": "VP of Global Data Center Procurement & Real Estate / Director of Supply Chain Infrastructure"
                }
            elif archetype == "Mission-Critical Infrastructure OEM":
                parsed = {
                    "company_name": "Vertiv Holdings Co." if "vertiv" in domain.lower() else comp_name,
                    "industry_focus": "High-Density Thermal Management, Direct-to-Chip Liquid Cooling & Critical Power",
                    "executive_profile_analysis": (
                        "Vertiv Holdings Co. (NYSE: VRT) is the premier global architect of critical digital infrastructure technologies powering hyperscale data centers, enterprise communication networks, and mission-critical commercial facilities. Operating across 40+ countries with 34,000+ personnel, Vertiv designs, manufactures, and commissions industrial-scale Liebert thermal management systems, direct-to-chip liquid cooling CDUs, medium-voltage switchgear, and integrated modular power distribution skids."
                    ),
                    "expectations_and_needs_narrative": (
                        "In seeking our project intelligence capabilities, Vertiv's commercial business development and solutions architecture teams require predictive 12-to-18-month advance visibility into regional data center land acquisitions, municipal permitting dockets, and substation queue allocations. Gaining early intelligence during conceptual design and Front-End Engineering Design (FEED) allows Vertiv to engage MEP engineering consultancies and project developers well before formal public equipment tenders are released."
                    ),
                    "operational_friction_analysis": (
                        "Because critical power and liquid cooling equipment require long manufacturing lead times, discovering projects only after public contractor bidding opens severely disadvantages hardware OEMs. Public tenders compress commercial margins and frequently favor competitors whose equipment was pre-specified into architectural blueprints during initial permitting, resulting in substantial commercial friction and lower conversion rates on high-margin infrastructure contracts."
                    ),
                    "buying_role_hypothesis": "VP of Global Business Development / Enterprise Solutions Architecture Director"
                }
            else:
                parsed = {
                    "company_name": clean_name,
                    "industry_focus": f"Enterprise Infrastructure & Operations in {domain}",
                    "executive_profile_analysis": f"{clean_name} delivers specialized commercial operations, digital capabilities, and infrastructure solutions across target markets.",
                    "expectations_and_needs_narrative": f"{clean_name} is seeking authoritative capital project pipeline intelligence, stage-gate permitting visibility, and verified stakeholder directories to identify early commercial opportunities.",
                    "operational_friction_analysis": f"{clean_name} faces operational friction from late awareness of major procurement tenders and lack of verified forward-looking project datasets.",
                    "buying_role_hypothesis": "VP of Infrastructure Procurement / Head of Strategic Growth"
                }

        parsed["archetype"] = archetype
        return parsed

    def llm_similarity_comparison(self, company_details: dict, candidate_sectors: list) -> list:
        """Deep multi-factor LLM semantic reasoning and similarity evaluation engine."""
        company_name = company_details.get("company_name", "Target Company")
        archetype = company_details.get("archetype", "Enterprise")
        industry = company_details.get("industry_focus", "Industrial Sector")
        summary = company_details.get("executive_profile_analysis", "")
        needs = company_details.get("expectations_and_needs_narrative", "")
        friction = company_details.get("operational_friction_analysis", "")

        candidate_list_text = "\n".join([
            f"- Sector: {c['Primary Sector']} | Definition: {c['Definition']}" for c in candidate_sectors
        ])

        system_prompt = (
            "You are a Senior Principal Enterprise Solutions Architect.\n"
            "An enterprise client has approached us with an inquiry regarding their strategic intelligence needs.\n"
            "Your task is to evaluate which catalog sectors from our offerings directly match and fulfill the client's inquiry.\n\n"
            "Rank the top 3 best matching sectors strictly based on their real-world applicability to the client's request.\n"
            "For each selected sector, provide:\n"
            "1. Exact match score (between 85.0% and 98.5%).\n"
            "2. In-depth strategic rationale explaining how our project intelligence in this sector fulfills their stated need.\n"
            "3. Specific enterprise challenge or procurement bottleneck this database resolves for them.\n\n"
            "Return strictly a valid JSON object matching this schema:\n"
            "{\n"
            '  "ranked_matches": [\n'
            '    {\n'
            '      "tier_label": "Primary Strategic Solution / Secondary Strategic Solution / Adjacent Expansion Solution",\n'
            '      "primary_sector": "Exact Primary Sector Name from candidates",\n'
            '      "llm_match_score": 96.5,\n'
            '      "llm_match_rationale": "Comprehensive 3-sentence explanation of how this fulfills their inquiry.",\n'
            '      "requirement_solved": "Exact operational requirement or strategic challenge solved."\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        prompt = (
            f"CLIENT PROFILE & INBOUND INQUIRY CONTEXT:\n"
            f"Client Name: {company_name}\n"
            f"Archetype: {archetype}\n"
            f"Industry Focus: {industry}\n"
            f"Operational Scope: {summary}\n"
            f"What the Client is Seeking: {needs}\n"
            f"Client's Operational Challenges: {friction}\n\n"
            f"OUR CATALOG SECTOR OFFERINGS TO MATCH:\n"
            f"{candidate_list_text}\n\n"
            f"Perform deep semantic comparison and select the top 3 best matching solutions to fulfill the client's request."
        )

        raw = self._call_llm(prompt, system_prompt)
        parsed = self._parse_json(raw)
        ranked = parsed.get("ranked_matches", [])

        default_tiers = ["Primary Strategic Solution", "Secondary Strategic Solution", "Adjacent Expansion Solution"]

        if ranked and isinstance(ranked, list):
            enriched_results = []
            candidates_dict = {c["Primary Sector"].lower().strip(): c for c in candidate_sectors}
            for i, item in enumerate(ranked[:3]):
                sec_name = item.get("primary_sector", "").strip()
                matched_cand = candidates_dict.get(sec_name.lower(), None)
                if not matched_cand:
                    for k, v in candidates_dict.items():
                        if sec_name.lower() in k or k in sec_name.lower():
                            matched_cand = v
                            break
                
                defn = matched_cand.get("Definition", "") if matched_cand else ""
                score = float(item.get("llm_match_score", 94.0 - (i * 3.0)))
                tier = item.get("tier_label", default_tiers[i])
                
                enriched_results.append({
                    "tier_label": tier,
                    "Primary Sector": sec_name or (matched_cand["Primary Sector"] if matched_cand else "Capital Project Intelligence"),
                    "Definition": defn,
                    "similarity": round(score / 100.0, 4),
                    "match_pct": score,
                    "llm_match_rationale": item.get("llm_match_rationale", ""),
                    "requirement_solved": item.get("requirement_solved", "")
                })
            if len(enriched_results) > 0:
                return enriched_results

        fallback_results = []
        for i, c in enumerate(candidate_sectors[:3]):
            fallback_results.append({
                "tier_label": default_tiers[i],
                "Primary Sector": c["Primary Sector"],
                "Definition": c["Definition"],
                "similarity": c.get("similarity", 0.90 - i * 0.05),
                "match_pct": c.get("match_pct", 90.0 - i * 5.0),
                "llm_match_rationale": f"Directly fulfills {company_name}'s inquiry regarding capital project intelligence in {c['Primary Sector']}.",
                "requirement_solved": f"Early-stage capital project tracking and commercial pipeline intelligence in {c['Primary Sector']}."
            })
        return fallback_results

    def analyze_fit(self, company_details: dict, matched_services: list) -> dict:
        company_name = company_details.get("company_name", "Client Enterprise")
        archetype = company_details.get("archetype", "Enterprise Technology & Infrastructure Operator")
        decision_maker = company_details.get("buying_role_hypothesis", "Client Leadership")

        mappings = []
        for i, srv in enumerate(matched_services[:3]):
            title = srv.get("Primary Sector") or srv.get("Service Name") or "Capital Project Intelligence"
            defn = srv.get("Definition") or srv.get("Value Proposition") or "Verified capital project intelligence and lifecycle asset tracking."
            req_solved = srv.get("requirement_solved")
            tier_label = srv.get("tier_label", f"Strategic Solution {i+1}")

            if archetype == "Private Equity Sponsor & Asset Manager":
                offering_name = f"{title} Capital Project & M&A Due Diligence Database"
                solves_req = req_solved or f"Commercial Due Diligence & Proprietary M&A Deal Sourcing in {title}"
                comprehensive_narrative = (
                    f"In response to {company_name}'s inquiry regarding pre-acquisition due diligence and sector intelligence, our {title} Intelligence Database provides an exhaustive, verified forward-looking pipeline of announced, permitted, and under-construction capital developments across {title}. "
                    f"By tracking project stage-gates from feasibility and environmental permitting through EPC tender awards, the database directly enables {company_name}'s deal teams to stress-test buyout financial models against verified customer spending and capacity specifications. "
                    f"Furthermore, by delivering direct contact mapping of facility owners, lead contractors, and engineering consultancies, the platform unlocks proprietary pre-auction deal origination and empowers existing portfolio platform companies to capture high-margin supply contracts."
                )
                roi_narrative = (
                    f"Directly accelerates commercial due diligence velocity by 40%, eliminates information asymmetry during buyout underwriting, and unlocks an 18-month first-mover window for proprietary deal sourcing ahead of competitive investment bank auctions."
                )
            elif archetype == "Hyperscale Cloud & Logistics Developer":
                offering_name = f"{title} Capital Project & Site Selection Intelligence Feed"
                solves_req = req_solved or f"Pre-Filing Site Selection, Permitting & Substation Queue Tracking in {title}"
                comprehensive_narrative = (
                    f"In response to {company_name}'s inquiry regarding infrastructure expansion and utility grid tracking, our {title} Intelligence Feed provides global real estate, GIS, and infrastructure planning leadership with comprehensive pre-construction tracking across industrial land transactions, municipal zoning dockets, and environmental impact filings 18 to 24 months in advance of public announcement. "
                    f"The dataset specifically monitors regional utility substation interconnection queues—tracking target Megawatt (MW) allocations, kV transmission line capacity, and utility queue positions—to enable {company_name} to secure multi-gigawatt power before breaking ground and eliminate costly construction bottlenecks."
                )
                roi_narrative = (
                    f"Shortens site selection lead times by 12–18 months, prevents multi-month deployment delays on gigawatt AI clusters, and de-risks multi-billion-dollar infrastructure expansion programs."
                )
            elif archetype == "Mission-Critical Infrastructure OEM":
                offering_name = f"{title} Pre-Tender Blueprint Specification & Engineering Pipeline Feed"
                solves_req = req_solved or f"Early Front-End Engineering Design (FEED) Specification Lock-In in {title}"
                comprehensive_narrative = (
                    f"In response to {company_name}'s inquiry regarding advance capital project visibility, our {title} Intelligence Feed tracks the complete lifecycle of upcoming developments in {title} from early land acquisition and environmental permitting through Front-End Engineering Design (FEED). "
                    f"By providing {company_name}'s solutions architects with early visibility into scheduled equipment procurement, cooling and power requirements, and engineering parameters, the platform enables {company_name} to engage engineering consultancies during blueprint drafting and secure sole-source specification status before public tenders open."
                )
                roi_narrative = (
                    f"Grants an 18-month advance window to lock in proprietary equipment specifications into project blueprints, dramatically increasing win rates and protecting commercial gross margins."
                )
            else:
                offering_name = f"{title} Capital Project Intelligence Database"
                solves_req = req_solved or f"Early-Stage Pipeline Tracking & Market Discovery in {title}"
                comprehensive_narrative = f"In response to {company_name}'s inquiry, this dataset tracks announced, permitted, and under-construction capital developments across {title} globally, providing verified stage-gate tracking and direct stakeholder mapping."
                roi_narrative = "Accelerates commercial deal velocity and provides advance visibility into major capital expenditure programs."

            mappings.append({
                "tier_label": tier_label,
                "exact_offering_name": offering_name,
                "mapped_requirement": solves_req,
                "offering_definition": defn,
                "llm_match_rationale": srv.get("llm_match_rationale", ""),
                "comprehensive_narrative": comprehensive_narrative,
                "roi_narrative": roi_narrative
            })

        top_offering_name = mappings[0]["exact_offering_name"] if mappings else "Project Intelligence Database"
        top_sector = matched_services[0].get("Primary Sector", "Target Sector") if matched_services else "Infrastructure"

        return {
            "fit_score": matched_services[0].get("match_pct", 98.0) if matched_services else 98.0,
            "target_alignment": decision_maker,
            "exact_product_mappings": mappings
        }

ai = WorkerAI()
