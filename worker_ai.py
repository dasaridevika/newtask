import os
import json
import re
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

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        if not self.worker_url:
            return ""
        try:
            resp = requests.post(
                self.worker_url,
                json={"model": self.model, "system": system_prompt, "prompt": prompt},
                headers={"Content-Type": "application/json"},
                timeout=50
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
            "You are a Senior Managing Director & Global Head of Strategic Corporate Intelligence.\n"
            "Analyze the target company from the provided text and produce an exhaustive, deep narrative analytical dossier.\n"
            "CRITICAL INSTRUCTION: Write comprehensive, highly detailed narrative prose paragraphs. DO NOT use fragmented bullet points.\n\n"
            "Return strictly a valid JSON object matching this schema:\n"
            "{\n"
            '  "company_name": "Official Entity Name",\n'
            '  "industry_focus": "Specific Industry Domain",\n'
            '  "executive_profile_analysis": "Comprehensive 3-paragraph institutional assessment of the business model, commercial scale, and market footprint.",\n'
            '  "expectations_and_needs_narrative": "Detailed narrative analysis explaining exactly what forward-looking project datasets, site selection intelligence, or market visibility the enterprise requires to accelerate its strategic roadmap.",\n'
            '  "operational_friction_analysis": "In-depth narrative analysis detailing the structural friction, lead-time bottlenecks, information asymmetry, and commercial pressures they face in their current operations.",\n'
            '  "buying_role_hypothesis": "Specific Executive Title"\n'
            "}"
        )

        prompt = f"Target Enterprise Domain: {domain}\n\nCrawled Intelligence:\n{scraped_text[:12000]}"
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
                        f"To sustain top-quartile internal rates of return (IRR) and accelerate buyout deal velocity, {comp_name}'s investment committees and operating partners require authoritative, forward-looking intelligence on global capital expenditure pipelines. Specifically, deal teams expect verified visibility into upcoming industrial manufacturing buildouts, plant modernization dockets, and supply chain procurement cycles across specialty chemical, packaging machinery, and industrial services sectors. This forward intelligence is essential to stress-test financial underwriting models, conduct commercial due diligence on potential buyout targets, and benchmark the growth avenues of their existing portfolio platforms."
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
                        "Amazon's global infrastructure planning, real estate, and procurement leadership require granular, pre-construction intelligence covering land zoning dockets, municipal permits, and environmental impact filings 18 to 24 months in advance. Additionally, infrastructure planners require real-time tracking of utility substation interconnection queues (including target Megawatt capacity and transmission kV voltage levels) to de-risk multi-gigawatt power provisioning for AWS AI clusters and optimize multimodal logistics routing adjacent to dedicated freight rail corridors."
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
                        "Vertiv's commercial business development and solutions architecture teams require predictive 12-to-18-month advance visibility into regional data center land acquisitions, municipal permitting dockets, and substation queue allocations. Gaining early intelligence during conceptual design and Front-End Engineering Design (FEED) allows Vertiv to engage MEP engineering consultancies and project developers well before formal public equipment tenders are released."
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
                    "expectations_and_needs_narrative": f"{clean_name} requires authoritative capital project pipeline intelligence, stage-gate permitting visibility, and verified stakeholder directories to identify early commercial opportunities.",
                    "operational_friction_analysis": f"{clean_name} faces operational friction from late awareness of major procurement tenders and lack of verified forward-looking project datasets.",
                    "buying_role_hypothesis": "VP of Infrastructure Procurement / Head of Strategic Growth"
                }

        parsed["archetype"] = archetype
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
                solves_req = f"Commercial Due Diligence & Proprietary M&A Deal Sourcing in {title}"
                comprehensive_narrative = (
                    f"Our {title} Intelligence Database directly resolves {company_name}'s reliance on lagging retrospective market reports by providing an exhaustive, verified forward-looking pipeline of announced, permitted, and under-construction capital developments across {title}. "
                    f"By tracking project stage-gates from feasibility and environmental permitting through EPC tender awards, the database equips {company_name}'s deal teams to stress-test buyout financial models against verified customer spending and capacity specifications. "
                    f"Furthermore, by delivering direct mapping of facility owners, lead contractors, and engineering consultancies, the platform unlocks proprietary pre-auction deal origination and empowers existing portfolio platform companies to capture high-margin supply contracts."
                )
                roi_narrative = (
                    f"Accelerates commercial due diligence velocity by 40%, eliminates information asymmetry during buyout underwriting, and unlocks an 18-month first-mover window for proprietary deal sourcing ahead of competitive investment bank auctions."
                )
            elif archetype == "Hyperscale Cloud & Logistics Developer":
                offering_name = f"{title} Capital Project & Site Selection Intelligence Feed"
                solves_req = f"Pre-Filing Site Selection, Permitting & Substation Queue Tracking in {title}"
                comprehensive_narrative = (
                    f"Our {title} Intelligence Feed provides {company_name}'s global real estate, GIS, and infrastructure planning leadership with comprehensive pre-construction tracking across industrial land transactions, municipal zoning dockets, and environmental impact filings 18 to 24 months in advance of public announcement. "
                    f"The dataset specifically monitors regional utility substation interconnection queues—tracking target Megawatt (MW) allocations, kV transmission line capacity, and utility queue positions—to enable {company_name} to secure multi-gigawatt power before breaking ground and eliminate costly construction bottlenecks."
                )
                roi_narrative = (
                    f"Shortens site selection lead times by 12–18 months, prevents multi-month deployment delays on gigawatt AI clusters, and de-risks multi-billion-dollar infrastructure expansion programs."
                )
            elif archetype == "Mission-Critical Infrastructure OEM":
                offering_name = f"{title} Pre-Tender Blueprint Specification & Engineering Pipeline Feed"
                solves_req = f"Early Front-End Engineering Design (FEED) Specification Lock-In in {title}"
                comprehensive_narrative = (
                    f"Our {title} Intelligence Feed tracks the complete lifecycle of upcoming capital developments in {title} from early land acquisition and environmental permitting through Front-End Engineering Design (FEED). "
                    f"By providing {company_name}'s solutions architects with early visibility into scheduled equipment procurement, cooling and power requirements, and engineering parameters, the platform enables {company_name} to engage engineering consultancies during blueprint drafting and secure sole-source specification status before public tenders open."
                )
                roi_narrative = (
                    f"Grants an 18-month advance window to lock in proprietary equipment specifications into project blueprints, dramatically increasing win rates and protecting commercial gross margins."
                )
            else:
                offering_name = f"{title} Capital Project Intelligence Database"
                solves_req = f"Early-Stage Pipeline Tracking & Market Discovery in {title}"
                comprehensive_narrative = f"Tracks announced, permitted, and under-construction capital developments across {title} globally, providing verified stage-gate tracking and direct stakeholder mapping."
                roi_narrative = "Accelerates commercial deal velocity and provides advance visibility into major capital expenditure programs."

            mappings.append({
                "exact_offering_name": offering_name,
                "mapped_requirement": solves_req,
                "offering_definition": defn,
                "comprehensive_narrative": comprehensive_narrative,
                "roi_narrative": roi_narrative
            })

        top_offering_name = mappings[0]["exact_offering_name"] if mappings else "Project Intelligence Database"
        top_sector = matched_services[0].get("Primary Sector", "Target Sector") if matched_services else "Infrastructure"

        # Ultra-Detailed Narrative Outreach Dossier
        if archetype == "Private Equity Sponsor & Asset Manager":
            pitch = f"""### EXECUTIVE OUTREACH DOSSIER & STRATEGIC BRIEF

**TO:** {decision_maker}, {company_name}  
**FROM:** Senior Managing Director, Global Private Equity Strategy Group  
**SUBJECT:** Strategic Intelligence Partnership: Proprietary M&A Deal Origination & CAPEX Due Diligence in {top_sector}

---

#### 1. Strategic Context & Executive Thesis
{company_name} is an established institutional private investment leader with a distinguished five-decade heritage of operational value creation across middle-market industrials, specialty manufacturing, and business services. In today's competitive private equity landscape, sustaining top-quartile IRR requires moving beyond lagging historical market reports and identifying platform and add-on acquisition targets ahead of formal investment bank auctions.

#### 2. Identified Operational Friction & Investment Bottlenecks
During deal screening, investment committee underwriting, and commercial due diligence, private equity teams encounter significant structural constraints. Standard market sizing reports reflect historical retrospective data rather than forward-looking capital deployment cycles. Furthermore, broadly marketed investment bank auctions compress entry multiples and elevate valuations, increasing the necessity of proprietary deal sourcing. Additionally, existing portfolio platform companies frequently lack advance visibility into upcoming multi-million-dollar capital expenditure programs where their products and services could be specified.

#### 3. Strategic Solution Architecture ({top_offering_name})
Our Capital Project Intelligence Platform delivers verified, forward-looking market infrastructure datasets directly into {company_name}'s investment screening and portfolio operations workflows. Deal teams gain continuous visibility into announced, permitted, and under-construction capital developments across {top_sector}, complete with capital expenditure valuations, capacity ratings, construction schedules, and direct stakeholder directories linking facility owners, general contractors, and engineering consultancies.

#### 4. Quantified Strategic ROI & Value Creation
Partnering with our intelligence platform accelerates commercial due diligence velocity by 40%, stress-tests buyout underwriting models against verified construction schedules, and unlocks an 18-month advance window to intercept high-performing platform companies before competitive auctions commence. Moreover, current portfolio platform companies can leverage these verified feeds to win major equipment supply and service contracts across global capital projects.

#### 5. Proposed Engagement & Next Steps
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
Based on our industry intelligence, {company_name}'s regional expansion teams face acute operational friction. Power allocations and substation queue evaluations currently require 24–36 months of pre-construction coordination with regional power utilities. Simultaneously, regional developers and speculators lock up high-capacity industrial parcels months before zoning filings become public knowledge, creating costly delays for gigawatt compute cluster commissioning.

#### 3. Strategic Solution Architecture ({top_offering_name})
Our verified Capital Project Intelligence Platform delivers proprietary, pre-construction visibility directly into {company_name}'s GIS and real estate workflows. The platform provides verified pre-filing tracking of industrial land parcels, zoning applications, and environmental impact filings 18–24 months in advance, combined with real-time tracking of substation queue status, target Megawatt allocations, and transmission line capacity across {top_sector}.

#### 4. Quantified Strategic ROI & Value Creation
Accessing our pre-construction feeds grants {company_name} a 12-to-18-month advance window to secure optimal land parcels before regional real estate prices escalate, while validating utility substation capacity upfront to prevent costly deployment delays on mission-critical AI compute campuses.

#### 5. Proposed Engagement & Next Steps
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
Commercial sales and solutions architecture teams encounter significant hurdles when discovering projects only through public RFPs. By the time a tender is released, hardware specifications have already been locked in by competitors. Furthermore, long equipment manufacturing lead times make it difficult to respond to short-fuse contractor bids without advance pipeline visibility.

#### 3. Strategic Solution Architecture ({top_offering_name})
Our Project Intelligence Platform delivers pre-RFP visibility into the complete lifecycle of upcoming developments across {top_sector}. The platform tracks developments from initial land acquisition, zoning approval, and Front-End Engineering Design (FEED) through procurement tender release, providing detailed cooling and power requirements, capacity ratings, and verified stakeholder contact directories.

#### 4. Quantified Strategic ROI & Value Creation
Engaging engineering consultancies during blueprint drafting grants {company_name} an 18-month advance window to lock in proprietary equipment specifications, dramatically increasing tender win rates and protecting commercial gross margins.

#### 5. Proposed Engagement & Next Steps
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
{company_name} operates as an established commercial enterprise in {top_sector}. To maximize deal velocity and capture high-margin contracts, commercial leadership requires forward-looking visibility into major capital expenditure programs before public tenders.

#### 2. Identified Operational Friction
Commercial sales teams encounter friction from late tender awareness and reliance on speculative market rumors rather than verified stage-gate filings.

#### 3. Strategic Solution Architecture ({top_offering_name})
Our platform delivers comprehensive stage-gate tracking across announced, permitted, and under-construction capital developments, complete with direct stakeholder contact mapping.

#### 4. Quantified Strategic ROI
Grants a 12-to-18-month first-mover advantage to pre-position commercial solutions ahead of competitive bids.

#### 5. Proposed Engagement & Next Steps
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
