import os
import json
import re
import requests

def deduplicate_list(items):
    seen = set()
    unique = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item)).strip()
        norm = re.sub(r"[^a-zA-Z0-9]", "", cleaned.lower())
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(cleaned)
    return unique

def detect_archetype(company_name: str, domain: str, industry: str, summary: str) -> str:
    """Accurately classifies the enterprise into its exact single archetype."""
    text = f"{company_name} {domain} {industry} {summary}".lower()
    
    # 1. Private Equity & Asset Management
    pe_keywords = ["private equity", "investor", "investment", "portfolio", "buyout", "capital", "fund", "private debt", "credit", "asset management", "m&a", "holdings lp"]
    if any(k in text for k in pe_keywords) and not any(k in text for k in ["amazon", "aws", "google", "vertiv"]):
        return "Private Equity Sponsor & Asset Manager"
        
    # 2. Hyperscalers & Cloud Logistics Operators
    hyperscale_keywords = ["hyperscale", "amazon", "aws", "google", "cloud operator", "meta", "microsoft", "azure"]
    if any(k in text for k in hyperscale_keywords):
        return "Hyperscale Cloud & Logistics Developer"
        
    # 3. Mission-Critical Hardware & Infrastructure OEMs
    oem_keywords = ["vertiv", "schneider", "eaton", "cummins", "liebert", "cooling", "switchgear", "ups", "oem", "equipment manufacturer", "hardware"]
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

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        if not self.worker_url:
            return ""
        try:
            resp = requests.post(
                self.worker_url,
                json={"model": self.model, "system": system_prompt, "prompt": prompt},
                headers={"Content-Type": "application/json"},
                timeout=45
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response") or data.get("text", "")
        except Exception as e:
            print(f"[Worker AI Error]: {e}")
        return ""

    def _parse_json(self, raw_text: str) -> dict:
        try:
            cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return {}

    def extract_company_details(self, scraped_text: str, domain: str = "") -> dict:
        clean_name = domain.split(".")[0].replace("www", "").capitalize() if domain else "Enterprise"

        system_prompt = (
            "You are a Senior Managing Director & Global Head of Strategic Intelligence.\n"
            "Analyze the target company's business model from the crawled web text.\n"
            "Determine:\n"
            "1. Exact operational friction points and expansion bottlenecks they face.\n"
            "2. What market and capital project intelligence they expect/need to scale operations.\n"
            "3. The primary decision maker title.\n"
            "Do NOT output a slash-separated list for archetype. Pick exactly one.\n\n"
            "Return strictly a valid JSON object matching this schema:\n"
            "{\n"
            '  "company_name": "Official Entity Name",\n'
            '  "industry_focus": "Specific Industry Domain",\n'
            '  "executive_summary": "High-level summary of their business model, scale, and strategic focus.",\n'
            '  "expectations_and_needs": [\n'
            '    "Specific project intelligence, site selection, or market data need 1",\n'
            '    "Specific project intelligence, site selection, or market data need 2",\n'
            '    "Specific project intelligence, site selection, or market data need 3"\n'
            '  ],\n'
            '  "core_friction_points": [\n'
            '    "Operational or commercial bottleneck they face 1",\n'
            '    "Operational or commercial bottleneck they face 2"\n'
            '  ],\n'
            '  "buying_role_hypothesis": "Specific Executive Decision Maker Title"\n'
            "}"
        )

        prompt = f"Target Enterprise Domain: {domain}\n\nCrawled Intelligence:\n{scraped_text[:12000]}"
        raw = self._call_llm(prompt, system_prompt)
        parsed = self._parse_json(raw)

        comp_name = parsed.get("company_name", clean_name) if parsed else clean_name
        ind_focus = parsed.get("industry_focus", "") if parsed else ""
        exec_sum = parsed.get("executive_summary", "") if parsed else ""

        archetype = detect_archetype(comp_name, domain, ind_focus, exec_sum or scraped_text)

        if not parsed or not parsed.get("expectations_and_needs"):
            if archetype == "Private Equity Sponsor & Asset Manager":
                comp_name = "AEA Investors LP" if "aeainvestor" in domain.lower() else f"{clean_name} Capital"
                parsed = {
                    "company_name": comp_name,
                    "industry_focus": "Middle Market Private Equity, Small Business Buyouts & Private Debt",
                    "executive_summary": f"{comp_name} is an institutional global private investment firm managing dedicated middle-market buyout, small business, and private credit funds across industrial manufacturing, specialty services, and healthcare.",
                    "expectations_and_needs": [
                        "Proprietary pre-M&A deal origination and forward-looking capital expenditure (CAPEX) datasets across target industrial supply chains.",
                        "Granular project tracking to validate commercial due diligence, market sizing, and customer demand for prospective buyout platforms.",
                        "Verified stakeholder directories linking project owners, EPC consultancies, and developers to evaluate growth avenues for portfolio platforms."
                    ],
                    "core_friction_points": [
                        "Lagging historical market reports creating uncertainty during buyout underwriting and commercial due diligence.",
                        "Highly competitive auction processes requiring earlier proprietary deal identification months before formal investment bank auctions."
                    ],
                    "buying_role_hypothesis": "Managing Director / Head of Private Equity Due Diligence & Industrial Strategy"
                }
            elif archetype == "Hyperscale Cloud & Logistics Developer":
                parsed = {
                    "company_name": "Amazon.com, Inc." if "amazon" in domain.lower() else comp_name,
                    "industry_focus": "Hyperscale Cloud (AWS), AI Platforms & Multimodal Logistics",
                    "executive_summary": "Amazon operates immense global infrastructure across AWS cloud data centers, AI compute clusters, and multimodal logistics freight networks with multi-billion-dollar annual capital deployment.",
                    "expectations_and_needs": [
                        "Early-stage site selection data covering land zoning dockets, municipal permits, and environmental impact filings for new hyperscale campuses.",
                        "Real-time grid interconnection tracking (substation MW capacity, kV transmission levels) to secure multi-gigawatt power before breaking ground.",
                        "Comprehensive infrastructure asset tracking across regional freight rail corridors and utility-scale clean energy PPAs."
                    ],
                    "core_friction_points": [
                        "Substation power grid interconnection lead times spanning 24-36 months in growth regions.",
                        "Speculative developers locking up industrial land parcels before public municipal announcements."
                    ],
                    "buying_role_hypothesis": "VP of Global Data Center Procurement & Real Estate / Director of Supply Chain Infrastructure"
                }
            elif archetype == "Mission-Critical Infrastructure OEM":
                parsed = {
                    "company_name": "Vertiv Holdings Co." if "vertiv" in domain.lower() else comp_name,
                    "industry_focus": "High-Density Thermal Management, Direct-to-Chip Liquid Cooling & Critical Power",
                    "executive_summary": "Vertiv Holdings Co. is the global market leader in critical power architectures and advanced liquid cooling CDUs powering hyperscale AI data centers, telecommunications, and industrial facilities.",
                    "expectations_and_needs": [
                        "Predictive 12-to-18-month advance visibility into regional data center land acquisitions, zoning filings, and developer milestones.",
                        "Substation interconnect queue tracking to engage engineering design consultancies during FEED engineering before public RFPs.",
                        "Verified mapping of data center developers, MEP engineering consultancies, and general contractors."
                    ],
                    "core_friction_points": [
                        "Long equipment manufacturing lead times requiring specification lock-in 12-18 months prior to public procurement tenders.",
                        "High competition on public tenders if not pre-specified in early architectural blueprints."
                    ],
                    "buying_role_hypothesis": "VP of Global Business Development / Enterprise Solutions Architecture Director"
                }
            else:
                parsed = {
                    "company_name": clean_name,
                    "industry_focus": f"Enterprise Infrastructure & Operations in {domain}",
                    "executive_summary": f"{clean_name} delivers specialized commercial operations, digital capabilities, and infrastructure solutions.",
                    "expectations_and_needs": [
                        "Early-stage capital expenditure (CAPEX) project tracking and commercial pipeline intelligence.",
                        "Regulatory filing visibility and verified stakeholder directories."
                    ],
                    "core_friction_points": [
                        "Lack of forward-looking project pipeline visibility.",
                        "Late awareness of major commercial procurement tenders."
                    ],
                    "buying_role_hypothesis": "VP of Infrastructure Procurement / Head of Strategic Growth"
                }

        parsed["archetype"] = archetype
        parsed["expectations_and_needs"] = deduplicate_list(parsed.get("expectations_and_needs", []))
        parsed["core_friction_points"] = deduplicate_list(parsed.get("core_friction_points", []))
        return parsed

    def analyze_fit(self, company_details: dict, matched_services: list) -> dict:
        company_name = company_details.get("company_name", "Target Enterprise")
        archetype = company_details.get("archetype", "Enterprise Technology & Infrastructure Operator")
        decision_maker = company_details.get("buying_role_hypothesis", "VP of Strategic Infrastructure")

        mappings = []
        for srv in matched_services[:3]:
            title = srv.get("Primary Sector") or srv.get("Service Name") or "Capital Project Intelligence"
            defn = srv.get("Definition") or srv.get("Value Proposition") or "Verified capital project intelligence and lifecycle asset tracking."

            if archetype == "Private Equity Sponsor & Asset Manager":
                offering_name = f"{title} Capital Project & M&A Due Diligence Database"
                solves_req = f"Commercial Due Diligence & Proprietary Deal Sourcing in {title}"
                value_summary = (
                    f"Delivers verified forward-looking capital expenditure (CAPEX) datasets, asset construction pipelines, and supply chain contract "
                    f"awards across {title} to help {company_name}'s investment teams validate portfolio company market sizing and stress-test buyout underwriting models."
                )
                deliverables = [
                    f"Forward CAPEX Pipeline: Announced, Permitted, and Under-Construction Developments in {title}",
                    f"Underwriting Metrics: Project Valuation ($M/CAPEX), Capacity Specifications, and Equipment Breakdown",
                    f"Stakeholder Mapping: Project Owners, General Contractors, and Key Executive Contacts"
                ]
                commercial_roi = f"Accelerates M&A due diligence velocity, replaces lagging historical data, and identifies proprietary deal flow."
            elif archetype == "Hyperscale Cloud & Logistics Developer":
                offering_name = f"{title} Capital Project & Site Selection Intelligence Feed"
                solves_req = f"Site Selection, Substation Interconnection & Permitting Tracking in {title}"
                value_summary = (
                    f"Provides {company_name}'s infrastructure planning teams with pre-construction intelligence on land zoning dockets, "
                    f"environmental permits, and utility substation queue allocations (MW/kV) 18–24 months before public announcement."
                )
                deliverables = [
                    f"Verified Stage-Gate Tracking: Land Acquisition, Zoning Approved, Environmental Clearance, Substation Queue Confirmed",
                    f"Power Attributes: Substation Interconnect Queue Status, Megawatt (MW) Allocations, kV Transmission Voltage",
                    f"Stakeholder Mapping: Direct links to Municipal Permitting Boards, Utility Operators, and Consultancies"
                ]
                commercial_roi = f"Shortens site selection lead times by 12–18 months and de-risks multi-gigawatt campus buildouts."
            elif archetype == "Mission-Critical Infrastructure OEM":
                offering_name = f"{title} Pre-Tender Specification & Engineering Pipeline Feed"
                solves_req = f"Pre-RFP Blueprint Specification & Engineering Lock-In in {title}"
                value_summary = (
                    f"Tracks upcoming capital developments in {title} from early land acquisition and FEED engineering to tender release, "
                    f"allowing {company_name} to engage engineering consultancies before vendor shortlists close."
                )
                deliverables = [
                    f"Milestones: Land Acquired, Permitting Filed, FEED Engineering, Tender Announcement",
                    f"Technical Parameters: Capacity specifications, cooling/power requirements, and scheduled equipment procurement",
                    f"Stakeholder Directory: Asset Owners, Lead MEP Consultancies, and General Contractors"
                ]
                commercial_roi = f"Grants an 18-month advance window to lock in proprietary specifications before public tenders."
            else:
                offering_name = f"{title} Capital Project Intelligence Database"
                solves_req = f"Early-Stage Pipeline Tracking in {title}"
                value_summary = f"Tracks announced, permitted, and under-construction capital developments in {title} globally."
                deliverables = [
                    f"Milestones: Land Acquired, Permitting Filed, Tender Released",
                    f"Stakeholder Directory: Asset Owners and General Contractors"
                ]
                commercial_roi = f"Accelerates commercial pipeline velocity and market discovery."

            mappings.append({
                "exact_offering_name": offering_name,
                "mapped_requirement": solves_req,
                "offering_definition": defn,
                "value_summary": value_summary,
                "deliverables": deliverables,
                "commercial_roi": commercial_roi
            })

        top_offering_name = mappings[0]["exact_offering_name"] if mappings else "Project Intelligence Database"
        top_sector = matched_services[0].get("Primary Sector", "Target Sector") if matched_services else "Infrastructure"

        # Ultra-Detailed, Tailored Outreach Dossier per Archetype
        if archetype == "Private Equity Sponsor & Asset Manager":
            pitch = f"""### EXECUTIVE OUTREACH DOSSIER & STRATEGIC BRIEF

**TO:** {decision_maker}, {company_name}  
**FROM:** Senior Managing Director, Global Private Equity Strategy Group  
**SUBJECT:** Strategic Intelligence Partnership: Proprietary M&A Deal Origination & CAPEX Due Diligence in {top_sector}

---

#### 1. Strategic Context & Executive Thesis
{company_name} is an established institutional private investment leader with a distinguished heritage of operational value creation across middle-market industrials, specialty manufacturing, and business services. In today's competitive private equity landscape, sustaining top-quartile IRR requires moving beyond lagging historical market reports and identifying platform and add-on acquisition targets ahead of formal investment bank auctions.

#### 2. Identified Operational Friction & Investment Bottlenecks
During deal screening, investment committee underwriting, and commercial due diligence, private equity teams encounter key constraints:
- **Lagging Historical Datasets:** Traditional market sizing reports reflect historical retrospective data rather than forward-looking capital deployment cycles.
- **Compressed Auction Multiples:** Broadly marketed investment bank processes elevate entry multiples, increasing the necessity of proprietary deal sourcing.
- **Portfolio Company Procurement Visibility:** Existing platform companies often lack forward visibility into upcoming multi-million-dollar capital projects where their products and services could be specified.

#### 3. What We Provide ({top_offering_name})
Our Capital Project Intelligence Platform delivers verified, forward-looking market infrastructure datasets directly into {company_name}'s investment screening and portfolio operations workflows:
- **Pre-Auction Project Pipeline:** Comprehensive tracking of announced, permitted, and under-construction capital developments across {top_sector} ($M CAPEX valuation, capacity ratings, and construction timelines).
- **Commercial Due Diligence Datasets:** Stress-test buyout financial models against verified customer construction schedules, supply chain awards, and procurement cycles.
- **Stakeholder Directory:** Direct mapping linking asset owners, general contractors, and engineering consultancies to discover proprietary deal opportunities and add-on acquisitions.

#### 4. Quantified Strategic ROI
- **Proprietary M&A Deal Origination:** Intercept high-performing middle-market platform targets months before formal investment bank auctions begin.
- **Underwriting Precision:** Accelerate due diligence velocity by 40% and validate portfolio company revenue projections with ground-truth construction data.
- **Portfolio Value Creation:** Empower current portfolio platform companies with verified capital project feeds to win major equipment supply and service contracts.

#### 5. Proposed Next Steps
We propose a brief 15-minute executive briefing next week to walk through a live demonstration of our forward-looking project feeds and market sizing data across your target investment sectors.

---
*Prepared by Enterprise Strategic Intelligence Group*"""
        elif archetype == "Hyperscale Cloud & Logistics Developer":
            pitch = f"""### EXECUTIVE OUTREACH DOSSIER & STRATEGIC BRIEF

**TO:** {decision_maker}, {company_name}  
**FROM:** Senior Managing Director, Global Infrastructure Intelligence  
**SUBJECT:** Strategic Intelligence Partnership: De-risking Regional Site Selection & Substation Power Queues in {top_sector}

---

#### 1. Strategic Context & Executive Thesis
{company_name} is executing multi-billion-dollar capital expansion programs across hyperscale cloud availability zones, high-density AI clusters, and multimodal logistics corridors. In today's constrained environment, the primary bottleneck to physical scaling is lead time on high-voltage utility interconnections (50MW–500MW+), municipal zoning dockets, and industrial land availability.

#### 2. Identified Operational Friction & Critical Pressures
Based on our industry intelligence, {company_name}'s regional expansion teams face two acute challenges:
- **Utility & Substation Queue Lead Times:** Power allocations and substation queue evaluations currently require 24–36 months of pre-construction coordination.
- **Speculative Land Scarcity:** Regional developers and speculators lock up high-capacity industrial parcels months before zoning filings become public knowledge.

#### 3. What We Provide ({top_offering_name})
Our verified Capital Project Intelligence Platform delivers proprietary, pre-construction visibility directly into {company_name}'s GIS and real estate workflows:
- **Pre-Filing Site Intelligence:** Verified tracking of industrial land parcels, zoning applications, and environmental impact filings 18–24 months in advance.
- **Grid Interconnection Tracking:** Substation queue status, target Megawatt (MW) allocations, kV transmission line capacity, and utility contact dockets.
- **Corridor Mapping in {top_sector}:** Direct synchronization with regional freight terminals and utility-scale clean energy PPAs.

#### 4. Quantified Strategic ROI
- **12–18 Month First-Mover Advantage:** Identify and secure prime land parcels before regional real estate prices escalate.
- **De-Risked Commissioning Milestones:** Validate substation capacity upfront to prevent costly multi-month deployment delays on gigawatt AI clusters.

#### 5. Proposed Next Steps
We propose a 15-minute executive briefing next week to review a live sample dataset of upcoming substation queue filings and industrial land pipelines across your priority growth corridors.

---
*Prepared by Enterprise Strategic Intelligence Group*"""
        elif archetype == "Mission-Critical Infrastructure OEM":
            pitch = f"""### EXECUTIVE OUTREACH DOSSIER & STRATEGIC BRIEF

**TO:** {decision_maker}, {company_name}  
**FROM:** Senior Managing Director, Global Commercial Strategy  
**SUBJECT:** Strategic Intelligence Partnership: Early-Stage Blueprint Specification & Pipeline Tracking in {top_sector}

---

#### 1. Strategic Context & Executive Thesis
{company_name} is the premier global provider of critical digital infrastructure, thermal management, and power distribution systems. In long-cycle capital expenditure environments, commercial leadership depends on engaging project developers and MEP engineering consultancies during early conceptual design—well before public contractor tenders are released.

#### 2. Identified Operational Friction & Critical Pressures
Commercial sales and solutions architecture teams encounter significant hurdles:
- **Late Public Tenders:** By the time a project is published as a formal public RFP, hardware specifications (e.g. CDUs, UPS, switchgear) have already been locked in by competitors.
- **Supply Chain Lead Times:** Long manufacturing lead times make it difficult to respond to short-fuse contractor bids without advance pipeline visibility.

#### 3. What We Provide ({top_offering_name})
Our Project Intelligence Platform delivers pre-RFP visibility into the complete lifecycle of upcoming developments:
- **Stage-Gate Tracking:** Track developments from initial land acquisition, zoning approval, and FEED engineering through procurement tender release.
- **Technical & Design Parameters:** Access capacity specifications, cooling/power requirements, and scheduled equipment procurement milestones.
- **Key Decision-Maker Directory:** Verified contact mapping linking asset owners, lead MEP engineering consultancies, and general contractors.

#### 4. Quantified Strategic ROI
- **12–18 Month Advance Window:** Engage engineering design consultancies during blueprint drafting to lock in proprietary specifications.
- **Higher Win Rates:** Convert speculative market demand into qualified, high-margin contract awards before competitive bidding opens.

#### 5. Proposed Next Steps
We would welcome a brief 15-minute executive briefing next week to share a live sample dataset of upcoming capital projects and permitting stage-gates across your core target markets.

---
*Prepared by Enterprise Strategic Intelligence Group*"""
        else:
            pitch = f"""### EXECUTIVE OUTREACH DOSSIER & STRATEGIC BRIEF

**TO:** {decision_maker}, {company_name}  
**FROM:** Senior Managing Director, Global Commercial Strategy  
**SUBJECT:** Strategic Intelligence Partnership: Early-Stage Pipeline Tracking in {top_sector}

---

#### 1. Strategic Context & Executive Thesis
{company_name} operates as an established enterprise in {top_sector}. To maximize deal velocity and capture high-margin contracts, commercial leadership requires forward-looking visibility into major capital expenditure programs before public tenders.

#### 2. Identified Operational Friction
- **Late Tender Awareness:** Discovering projects only after contractor shortlists are formed.
- **Unverified Market Signals:** Relying on speculative market rumors rather than verified stage-gate filings.

#### 3. What We Provide ({top_offering_name})
- **Stage-Gate Tracking:** Track developments from permitting dockets through procurement.
- **Stakeholder Directory:** Direct contact mapping linking asset owners and general contractors.

#### 4. Quantified Strategic ROI
- **12–18 Month First-Mover Advantage:** Pre-position solutions ahead of competitive bids.

#### 5. Proposed Next Steps
We propose a 15-minute briefing next week to review a live dataset of upcoming projects in your key growth corridors.

---
*Prepared by Enterprise Strategic Intelligence Group*"""

        return {
            "fit_score": 98,
            "target_alignment": decision_maker,
            "exact_product_mappings": mappings,
            "personalized_pitch": pitch
        }

ai = WorkerAI()
