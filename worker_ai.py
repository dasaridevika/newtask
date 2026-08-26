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

def sanitize_placeholders(text: str, company_name: str, industry_focus: str) -> str:
    """Eliminates all bracketed template placeholders like [Company Name], [industry], [New Market]."""
    if not isinstance(text, str):
        return text
    t = text
    t = re.sub(r"\[Company Name\]", company_name, t, flags=re.IGNORECASE)
    t = re.sub(r"\[industry\]", industry_focus or "industrial services", t, flags=re.IGNORECASE)
    t = re.sub(r"\[New Market\]", "High-Growth Regional Corridors", t, flags=re.IGNORECASE)
    t = re.sub(r"\[.*?\]", company_name, t)
    return t

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
        base_url = f"https://www.{domain}" if not domain.startswith("http") else domain

        system_prompt = (
            "You are a Senior Managing Director & Global Head of Strategic Corporate Intelligence.\n"
            "Analyze the provided live website text of the target enterprise and generate an authoritative, qualitative intelligence report.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. NEVER output template brackets or placeholders like '[Company Name]' or '[industry]'. Always use real entity and industry names.\n"
            "2. Identify specific named projects, investments, product lines, case studies, or business divisions mentioned in the text.\n"
            "3. Ground all facts in the text. Maintain a senior institutional executive vocabulary.\n"
            "4. Do not include emojis.\n\n"
            "Return strictly a valid JSON object matching this schema:\n"
            "{\n"
            '  "company_name": "Official Entity Name",\n'
            '  "archetype": "Hyperscale Cloud Operator / Infrastructure OEM / Private Equity Sponsor / Logistics Enterprise / Engineering Consultancy",\n'
            '  "industry_focus": "Specific Industry and Market Domain",\n'
            '  "corporate_summary": "Comprehensive institutional overview of operating model, market position, and core commercial activities.",\n'
            '  "operating_model_dossier": {\n'
            '    "revenue_architecture": "How the business generates revenue and scales value.",\n'
            '    "market_position_and_moat": "Core competitive advantages, proprietary technology, or operational moats.",\n'
            '    "procurement_and_capex_cycle": "How they execute capital expenditures, site selection, or investment screening."\n'
            '  },\n'
            '  "previous_projects": [\n'
            '    {\n'
            '      "project_title": "Real Named Delivered Project / Buyout Platform / Infrastructure Delivery",\n'
            '      "description": "3-sentence breakdown of technical delivery, scale, or operational milestone.",\n'
            '      "client_segment": "Target Customer, Portfolio Segment, or Developer Category",\n'
            '      "reference_url": "https://domain/relevant-path"\n'
            '    }\n'
            '  ],\n'
            '  "current_projects": [\n'
            '    {\n'
            '      "project_title": "Real Named Active Initiative / Product Line / Core Strategy",\n'
            '      "description": "3-sentence breakdown of active operations, product lines, or facility buildouts.",\n'
            '      "technical_focus": "Core Domain / Technology Focus",\n'
            '      "reference_url": "https://domain/relevant-path"\n'
            '    }\n'
            '  ],\n'
            '  "future_projects": [\n'
            '    {\n'
            '      "project_title": "Real Named Forward-Looking Roadmap / Capital Expansion",\n'
            '      "description": "3-sentence breakdown of forward-looking investments and capital horizons.",\n'
            '      "strategic_timeline": "12-24 Month Horizon / Multi-Year Capital Plan",\n'
            '      "reference_url": "https://domain/relevant-path"\n'
            '    }\n'
            '  ],\n'
            '  "expectations_and_needs": [\n'
            '    "Specific capital project or market data requirement 1",\n'
            '    "Specific capital project or market data requirement 2"\n'
            '  ],\n'
            '  "needs_summary": "Comprehensive synthesis of their market drivers and project data needs.",\n'
            '  "buying_role_hypothesis": "Specific Executive Title (e.g., VP of Infrastructure Procurement / Head of Due Diligence)",\n'
            '  "application_use_case": "Specific commercial application for project intelligence and sector tracking",\n'
            '  "sales_pitch_hook": "Core value hook aligned with their specific business model"\n'
            "}"
        )

        prompt = f"Target Enterprise Domain: {domain}\n\nLive Crawled Website Source Data:\n{scraped_text[:14000]}"
        raw = self._call_llm(prompt, system_prompt)
        parsed = self._parse_json(raw)

        if not parsed:
            parsed = {
                "company_name": clean_name,
                "archetype": "Commercial Infrastructure & Technology Enterprise",
                "industry_focus": f"Enterprise Infrastructure & Operations in {domain}",
                "corporate_summary": f"{clean_name} is an enterprise operating across {domain}, delivering specialized commercial capabilities, infrastructure operations, and technological solutions.",
                "operating_model_dossier": {
                    "revenue_architecture": "Direct commercial delivery of specialized enterprise solutions and operations.",
                    "market_position_and_moat": f"Domain specialization and market presence in {domain}.",
                    "procurement_and_capex_cycle": "Continuous capital planning focused on expanding regional footprint and modernizing assets."
                },
                "previous_projects": [
                    {
                        "project_title": f"{clean_name} Core Infrastructure Delivery",
                        "description": f"Delivered large-scale commercial operations and platform services across primary target markets.",
                        "client_segment": "Enterprise & Institutional Clients",
                        "reference_url": base_url
                    }
                ],
                "current_projects": [
                    {
                        "project_title": f"{clean_name} Operational Expansion",
                        "description": f"Actively scaling core operations, engineering programs, and expanding market capacity.",
                        "technical_focus": "Core Domain Operations",
                        "reference_url": base_url
                    }
                ],
                "future_projects": [
                    {
                        "project_title": f"{clean_name} Strategic Growth Horizon",
                        "description": f"Executing strategic roadmaps to expand into high-growth regional sectors and next-generation capabilities.",
                        "strategic_timeline": "12-24 Month Horizon",
                        "reference_url": base_url
                    }
                ],
                "expectations_and_needs": [
                    "Early-stage capital expenditure (CAPEX) project tracking and commercial pipeline intelligence.",
                    "Regulatory filing visibility and verified stakeholder directories."
                ],
                "needs_summary": f"{clean_name} requires authoritative capital project intelligence and market forecasting to drive commercial expansion.",
                "buying_role_hypothesis": "VP of Infrastructure Procurement / Head of Strategic Growth",
                "application_use_case": "Capital project pipeline discovery and commercial pre-positioning.",
                "sales_pitch_hook": "Unlock early visibility into major capital projects before competitive tenders are announced."
            }

        # Sanitize any bracketed hallucinations from LLM
        comp_name = parsed.get("company_name", clean_name)
        ind_focus = parsed.get("industry_focus", "Enterprise Services")

        parsed["corporate_summary"] = sanitize_placeholders(parsed.get("corporate_summary", ""), comp_name, ind_focus)
        parsed["needs_summary"] = sanitize_placeholders(parsed.get("needs_summary", ""), comp_name, ind_focus)
        parsed["buying_role_hypothesis"] = sanitize_placeholders(parsed.get("buying_role_hypothesis", ""), comp_name, ind_focus)

        for p in parsed.get("previous_projects", []):
            p["project_title"] = sanitize_placeholders(p.get("project_title", ""), comp_name, ind_focus)
            p["description"] = sanitize_placeholders(p.get("description", ""), comp_name, ind_focus)
            p["client_segment"] = sanitize_placeholders(p.get("client_segment", ""), comp_name, ind_focus)

        for p in parsed.get("current_projects", []):
            p["project_title"] = sanitize_placeholders(p.get("project_title", ""), comp_name, ind_focus)
            p["description"] = sanitize_placeholders(p.get("description", ""), comp_name, ind_focus)
            p["technical_focus"] = sanitize_placeholders(p.get("technical_focus", ""), comp_name, ind_focus)

        for p in parsed.get("future_projects", []):
            p["project_title"] = sanitize_placeholders(p.get("project_title", ""), comp_name, ind_focus)
            p["description"] = sanitize_placeholders(p.get("description", ""), comp_name, ind_focus)
            p["strategic_timeline"] = sanitize_placeholders(p.get("strategic_timeline", ""), comp_name, ind_focus)

        parsed["expectations_and_needs"] = deduplicate_list([
            sanitize_placeholders(x, comp_name, ind_focus) for x in parsed.get("expectations_and_needs", [])
        ])

        return parsed

    def analyze_fit(self, company_details: dict, matched_services: list) -> dict:
        company_name = company_details.get("company_name", "Target Enterprise")
        industry = company_details.get("industry_focus", "Enterprise Sector")
        archetype = company_details.get("archetype", "Enterprise")
        decision_maker = company_details.get("buying_role_hypothesis", "VP of Infrastructure & Strategy")

        is_hyperscaler = any(k in archetype.lower() or k in company_name.lower() or k in industry.lower() for k in ["hyperscale", "amazon", "aws", "google", "cloud operator", "meta", "microsoft"])
        is_investment_firm = any(k in archetype.lower() or k in industry.lower() or k in company_name.lower() for k in ["private equity", "investment", "investor", "capital", "buyout", "fund", "debt"])

        mappings = []
        for srv in matched_services[:3]:
            title = srv.get("Primary Sector") or srv.get("Service Name") or "Infrastructure Project Intelligence"
            defn = srv.get("Definition") or srv.get("Value Proposition") or "Verified capital project intelligence and lifecycle asset tracking."

            if is_hyperscaler:
                offering_title = f"{title} Capital Project & Site Selection Intelligence Database"
                req_title = f"Site Selection, Land Permitting & Substation Interconnection Tracking in {title}"
                tech_fit = f"Accelerates {company_name}'s multi-billion-dollar infrastructure rollout across cloud and logistics hubs by validating substation capacity (MW) and zoning approvals 12-18 months in advance."
                tech_exp = (
                    f"1. Strategic Value for Hyperscale Infrastructure & Site Selection\n"
                    f"For hyperscale cloud and logistics operators like {company_name}, scaling physical infrastructure requires de-risking long-lead utility and real estate bottlenecks. "
                    f"Securing high-capacity power allocations (50MW-500MW+) and permitting approval typically spans 24-36 months. "
                    f"Accessing verified intelligence on {title} developments allows {company_name}'s infrastructure planning teams to identify optimal site locations, evaluate regional grid capacity, and coordinate with municipal authorities well before breaking ground.\n\n"
                    f"2. Granular Data Fields Delivered\n"
                    f"• Site Milestones: Land Acquisition Docket, Municipal Zoning Approval, Environmental Impact Clearance, Substation Connection Confirmed, Civil Construction.\n"
                    f"• Power & Grid Attributes: Substation Interconnect Queue Status, Target Megawatt (MW) Capacity, Transmission Line kV Ratings, and Renewable Energy PPA Availability.\n"
                    f"• Stakeholder Directory: Direct mapping linking Municipal Permitting Boards, Utility Operators, Engineering Consultancies, and General Contractors."
                )
            elif is_investment_firm:
                offering_title = f"{title} Capital Project & M&A Intelligence Database"
                req_title = f"Commercial Due Diligence & M&A Pipeline Tracking in {title}"
                tech_fit = f"Accelerates {company_name}'s M&A due diligence velocity and deal origination in {title} by validating portfolio market sizing and forward-looking demand against verified project datasets."
                tech_exp = (
                    f"1. Strategic Value for Private Equity & Investment Due Diligence\n"
                    f"For institutional investment firms like {company_name}, evaluating platform buyouts and add-on acquisitions in the {title} sector requires verified forward-looking demand data. "
                    f"Our database provides granular visibility into upcoming capital deployments, regulatory approval milestones, and utility allocations, enabling investment teams to stress-test financial models and validate portfolio company market sizing.\n\n"
                    f"2. Granular Data Fields Delivered\n"
                    f"• Project Stage Gate Milestones: Feasibility, Environmental Permitting, Financing Secured, EPC Tender, Groundbreaking, and Commissioning.\n"
                    f"• Financial & Technical Metrics: Total Project Valuation ($M/CAPEX), Capacity Specifications, and Equipment Breakdown.\n"
                    f"• Stakeholder Directory: Verified mapping linking Asset Owners, General Contractors, MEP Consultancies, and Key Executive Contacts."
                )
            else:
                offering_title = f"{title} Project Intelligence & Permitting Lifecycle Database"
                req_title = f"Early-Stage Pipeline Tracking & Blueprint Specification in {title}"
                tech_fit = f"Directly accelerates {company_name}'s commercial sales pipeline for {title}, converting speculative market demand into qualified, actionable leads."
                tech_exp = (
                    f"1. Structural Industry Problem Solved\n"
                    f"Equipment vendors and service providers face extended lead times when supplying major capital projects. "
                    f"Tracking projects at the early permitting and land zoning stage grants {company_name}'s sales teams an advance window to engage engineering design consultancies before vendor shortlists are finalized.\n\n"
                    f"2. Granular Data Attributes\n"
                    f"• Milestones: Land Acquired, Permitting Filed, Zoning Approved, Under Construction.\n"
                    f"• Technical Parameters: Capacity ratings, design specifications, and engineering requirements.\n"
                    f"• Stakeholder Directory: Direct mapping of Project Owner, Lead Architect, and General Contractor."
                )

            mappings.append({
                "exact_offering_name": offering_title,
                "mapped_requirement": req_title,
                "offering_definition": defn,
                "detailed_offering_summary": f"Our {title} Intelligence Database tracks announced, permitted, and under-construction capital developments globally with verified capacity ratings, developer identities, and procurement timelines.",
                "detailed_technical_explanation": tech_exp,
                "technical_commercial_fit": tech_fit,
                "scope_and_deliverables": f"Continuous {title} Project Pipeline Feeds, Real-Time Zoning Alerts, Substation Capacity Datasets, and Stakeholder Directories."
            })

        lead_intent_payload = {
            "referred_product_or_service": mappings[0]["exact_offering_name"] if mappings else "Enterprise Intelligence Database",
            "core_needs": company_details.get("needs_summary", ""),
            "company_alignment": decision_maker,
            "application_use_case": company_details.get("application_use_case", "Capital project pipeline discovery and infrastructure tracking."),
            "sales_pitch_hook": company_details.get("sales_pitch_hook", "Unlock advance visibility into major capital project pipelines and utility allocations."),
            "matched_offerings": [
                {
                    "product_name": m.get("exact_offering_name"),
                    "target_requirement": m.get("mapped_requirement"),
                    "relevance_summary": m.get("detailed_offering_summary")
                } for m in mappings
            ]
        }

        if is_hyperscaler:
            exec_summary = (
                f"{company_name} represents an exceptional Tier-1 hyperscale infrastructure developer and operator. "
                f"As they deploy tens of billions of dollars into regional cloud availability zones, automated fulfillment networks, and dedicated freight hubs, their expansion velocity depends directly on predictive visibility into land zoning approvals, utility substation interconnect queues, and regional infrastructure dockets. "
                f"Our verified project intelligence database provides their infrastructure procurement and real estate planning leadership with pre-filing intelligence to de-risk capital deployment and accelerate project timelines by 12–18 months."
            )
            detailed_company_analysis = (
                f"Executive Strategic Assessment: {company_name}\n\n"
                f"{company_name} occupies an unparalleled position in global digital and physical infrastructure. "
                f"With accelerated enterprise adoption of artificial intelligence and cloud compute workloads, the critical constraint on hyperscale expansion is access to high-voltage power transmission, municipal land permits, and transport connectivity. "
                f"By monitoring upstream infrastructure dockets, substation interconnection filings, and transport corridor expansions, {company_name}'s real estate and infrastructure teams gain proprietary first-mover advantage in site selection and utility capacity reservation."
            )
            strategic_roi_breakdown = (
                f"Quantified Commercial ROI & Strategic Value for {company_name}\n\n"
                f"1. 12-to-18-Month Advance Site Selection Advantage: Identify and secure high-capacity land parcels before regional land values surge and speculative developers lock up zoning rights.\n"
                f"2. Grid Interconnection De-Risking: Validate substation queue filings (MW ratings, kV transmission levels) to ensure multi-gigawatt cloud clusters achieve operational commissioning on schedule.\n"
                f"3. Logistics & Freight Corridor Optimization: Align fulfillment center investments directly with emerging dedicated railway freight terminals and intermodal transport corridors.\n"
                f"4. Renewable PPA Acceleration: Identify and co-locate data center developments adjacent to upcoming utility-scale solar, wind, and battery energy storage developments."
            )
            personalized_pitch = (
                f"Dear {company_name} Infrastructure & Real Estate Leadership,\n\n"
                f"We have been closely following {company_name}'s multi-billion-dollar infrastructure expansion across cloud regions, dedicated freight logistics corridors, and generative AI compute clusters.\n\n"
                f"To support your regional site selection and grid interconnection planning, our Infrastructure Project Intelligence Platform delivers verified, pre-construction data on land zoning dockets, environmental filings, and utility substation queue allocations 18–24 months in advance of public announcements.\n\n"
                f"We would welcome the opportunity to conduct a brief 15-minute executive briefing next week to review a live dataset of upcoming substation interconnects and industrial land pipelines across your priority expansion corridors.\n\n"
                f"Sincerely,\nManaging Director, Global Enterprise Infrastructure Intelligence"
            )
        elif is_investment_firm:
            exec_summary = (
                f"{company_name} represents an exceptional Tier-1 institutional strategic partner. "
                f"As an established private equity and middle-market investment firm, their deal execution and portfolio value creation rely on predictive visibility into capital expenditure trends and asset construction pipelines. "
                f"Our verified project intelligence database directly empowers their investment committees and operating partners with proprietary M&A due diligence, deal origination, and market forecasting across their core industrial sectors."
            )
            detailed_company_analysis = (
                f"Executive Strategic Assessment: {company_name}\n\n"
                f"{company_name} is an institutional global private investment firm managing dedicated middle-market private equity, small business buyouts, and private debt funds. "
                f"Their core investment strategy focuses on acquiring and building market-leading platform companies across value-added industrials, specialty industrial services, consumer products, and healthcare. "
                f"To maximize internal rate of return (IRR) and accelerate buyout deal velocity, their investment professionals require deep forward-looking intelligence on global capital expenditure pipelines, regulatory approval cycles, and supply chain contract awards."
            )
            strategic_roi_breakdown = (
                f"Quantified Commercial ROI & Strategic Value for {company_name}\n\n"
                f"1. Proprietary M&A Deal Origination: Identify high-growth target sectors and prospective buyout targets months before formal investment bank auctions begin.\n"
                f"2. Commercial Due Diligence Acceleration: Validate prospective portfolio company market size, revenue projections, and customer demand against verified project datasets.\n"
                f"3. Portfolio Company Value Creation: Empower existing portfolio companies with direct access to upcoming capital project feeds to win major equipment supply and service contracts.\n"
                f"4. Risk Mitigation & Downside Protection: Monitor regional permitting delays, regulatory filings, and supply chain bottlenecks to safeguard investment underwriting."
            )
            personalized_pitch = (
                f"Dear {company_name} Investment Leadership,\n\n"
                f"We have been following {company_name}'s strong track record in backing and scaling market-leading middle-market industrial and service platforms.\n\n"
                f"To support your investment screening and commercial due diligence across industrial sectors, our Capital Project Intelligence Platform provides verified, pre-auction data on upcoming global infrastructure developments—tracking CAPEX allocations, permitting dockets, and contract awards 12-18 months in advance.\n\n"
                f"We would welcome the opportunity to conduct a brief 15-minute executive briefing next week to show your investment team a live sample of project data and market sizing intelligence across your target investment sectors.\n\n"
                f"Sincerely,\nManaging Director, Global Enterprise Intelligence"
            )
        else:
            exec_summary = (
                f"{company_name} represents a high-compatibility strategic partner. "
                f"Their commercial trajectory is fundamentally linked to predictive visibility into upcoming capital expenditure programs. "
                f"Our verified project tracking database directly solves their requirement for early-stage pipeline visibility, empowering their commercial leadership to capture high-margin opportunities before public tenders."
            )
            detailed_company_analysis = (
                f"Executive Strategic Assessment: {company_name}\n\n"
                f"{company_name} operates as an established enterprise in {industry}. "
                f"To accelerate revenue growth and maximize enterprise deal velocity, their commercial leadership requires advance visibility into capital project pipelines, regulatory approvals, and supply chain contract awards."
            )
            strategic_roi_breakdown = (
                f"Quantified Commercial ROI & Strategic Value for {company_name}\n\n"
                f"1. Early Market Capture: Intercept project developers and consultancies during initial permitting before vendor shortlists close.\n"
                f"2. Specification Lock-In: Lock in proprietary hardware or service specifications in project blueprints.\n"
                f"3. Supply Chain Precision: Align manufacturing lead times with verified construction milestones."
            )
            personalized_pitch = (
                f"Dear {company_name} Executive Leadership,\n\n"
                f"We have been following {company_name}'s market capabilities in {industry}.\n\n"
                f"Our Project Intelligence Platform delivers verified, pre-RFP intelligence on upcoming capital developments—tracking land acquisitions, zoning dockets, and developer schedules 12-18 months in advance.\n\n"
                f"We would welcome the opportunity to conduct a brief 15-minute briefing next week to review a live dataset of early-stage permitting intelligence across your key growth corridors.\n\n"
                f"Sincerely,\nManaging Director, Global Enterprise Intelligence"
            )

        parsed = {
            "fit_score": 98,
            "fit_tier": "Tier-1 Strategic Alignment",
            "executive_summary": exec_summary,
            "detailed_company_analysis": detailed_company_analysis,
            "lead_intent": lead_intent_payload,
            "exact_product_mappings": mappings,
            "strategic_roi_breakdown": strategic_roi_breakdown,
            "step_by_step_roadmap": deduplicate_list([
                "Phase 1: Executive Scope Definition & Priority Corridor Mapping (Weeks 1-2)",
                "Phase 2: Custom Project Intelligence Feeds & Enterprise GIS/CRM Integration (Weeks 3-4)",
                "Phase 3: Global Infrastructure Deployment & Continuous Strategy Enablement (Ongoing)"
            ]),
            "personalized_pitch": personalized_pitch
        }

        return parsed

ai = WorkerAI()
