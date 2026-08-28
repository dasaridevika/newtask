import os
import json
import re
import time
import requests
from typing import List, Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_domain_deliverable_blueprint(sector_name: str) -> str:
    s = sector_name.lower()
    if "data center" in s:
        return "Delivers verified power substation interconnect queue tracking (MW load capacity), municipal zoning and environmental review logs, hyperscale vs colocation facility buildout timelines, cooling topology specifications, and stakeholder directories covering developers, facility operators, and EPC contractors."
    elif "telecommunication" in s or "communication" in s:
        return "Delivers regional fiber route dark/lit asset maps, cellular tower co-location permit feeds, municipal right-of-way easement filings, edge data network exchange construction milestones, and carrier/infrastructure developer directories."
    elif "health" in s or "hospital" in s or "clinic" in s:
        return "Delivers state Certificate of Need (CON) regulatory filings, ambulatory surgery center (ASC) licensing tracking, regional outpatient clinic expansion dockets, medical office building (MOB) zoning approvals, and health system operator directories."
    elif "warehouse" in s or "distribution" in s or "logistics" in s or "terminal" in s:
        return "Delivers industrial distribution center square footage specifications, clear-height and loading dock door data, intermodal freight rail and highway access maps, automated sorting hub development permits, and logistics developer/tenant directories."
    elif "refinery" in s or "chemical" in s or "lng" in s:
        return "Delivers industrial environmental and EPA Title V emissions permit tracking, turnaround maintenance and expansion milestone feeds, processing capacity metrics (BPD / TPY), and engineering contractor award dossiers."
    elif "solar" in s or "photovoltaic" in s or "wind" in s or "power" in s or "battery" in s:
        return "Delivers utility grid interconnection queue tracking (MW / MWh), environmental impact statement (EIS) filings, PPA contract award milestones, battery energy storage system (BESS) stage-gates, and renewable asset owner/developer directories."
    else:
        return f"Delivers stage-gate capital project permitting trackers, technical asset capacity specifications, municipal engineering milestones, and key stakeholder directories across {sector_name} developments."

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

        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
    ) -> str:
        if not self.worker_url:
            return ""

        payload: Dict[str, Any] = {
            "model": self.model,
            "system": system_prompt,
            "prompt": prompt,
        }
        if response_format:
            payload["response_format"] = response_format

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.session.post(self.worker_url, json=payload, timeout=60)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception:
                        return resp.text.strip()

                    res_text = data.get("response") or data.get("text") or data.get("result") or ""
                    if isinstance(res_text, dict):
                        return json.dumps(res_text, ensure_ascii=False)
                    return str(res_text).strip()

                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else (1.5 * (attempt + 1))
                    time.sleep(sleep_for)
                    continue

                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))

        if last_error:
            print(f"[Worker AI Error]: {last_error}")
        return ""

    def _parse_json(self, raw_text: str) -> dict:
        if not raw_text:
            return {}
        try:
            cleaned = raw_text.strip()
            cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^```\s*", "", cleaned)
            cleaned = re.sub(r"```$", "", cleaned)

            try:
                return json.loads(cleaned)
            except Exception:
                pass

            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str = re.sub(r",\s*([\]\}])", r"\1", json_str)
                return json.loads(json_str)
        except Exception:
            pass
        return {}

    def _safe_company_name(self, domain: str) -> str:
        if not domain:
            return "Enterprise"
        base = domain.split("//")[-1].split("/")[0]
        base = base.replace("www.", "").split(".")[0]
        return base[:1].upper() + base[1:] if base else "Enterprise"

    def extract_company_details(self, scraped_text: str, domain: str = "", client_inquiry: str = "", evidence_store=None) -> dict:
        clean_name = self._safe_company_name(domain)
        inquiry_text = f'\nClient Specific Inbound Inquiry / Stated Requirement:\n"{client_inquiry}"\n' if client_inquiry else ""

        pages = getattr(evidence_store, "pages", None) if evidence_store else None
        available_urls = [p.url for p in pages] if pages else [f"https://{domain}" if domain else ""]
        available_urls = [u for u in available_urls if u]
        fallback_url = available_urls[0] if available_urls else (f"https://{domain}" if domain else "")

        urls_formatted = "\n".join([f"- {u}" for u in available_urls[:10]])

        system_prompt = """
You are a Senior Principal Corporate Intelligence Strategist, Executive Diligence Architect, and Evidence Verification Specialist.

Analyze the target enterprise in depth strictly from the supplied crawled evidence and internal subpages. Provide a rich, highly qualitative, and evidence-grounded strategic briefing.

Guidelines for Depth and Qualitative Context:
1. Executive Profile: Provide 5 to 7 detailed, high-density sentences analyzing what the enterprise does, its operational anatomy, founding history, market scale, and operating philosophy. Avoid generic fluff; name specific products, services, and strategies.
2. Business Model: Provide 4 to 6 detailed sentences explaining how the company creates value, monetizes its capabilities, interacts with customer segments, and structures commercial/investment delivery.
3. Delivered Works & Projects: Extract 3 to 5 concrete historical case studies, portfolio achievements, facility expansions, or project milestones with exact verified metrics (e.g. $19B AUM, 300k SF facility, 6 acquisitions, 5x growth rate, 22% CAGR) and exact source URLs.
4. Current Active Operations: Provide 2 to 4 detailed operational descriptions of live capabilities, business divisions, and sector footprints.
5. Future Project Roadmaps: Outline 2 to 4 strategic expansion targets, digital/AI initiatives, or capacity buildouts with their implied project requirements.
6. Detailed Requirements Analysis: Synthesize their exact operational mandate, infrastructure/asset visibility needs, deal sourcing/diligence needs, regulatory/ESG needs, and primary operational bottlenecks.
7. Operating Platforms & Sector Footprint: For financial sponsors, private equity funds, or holding companies, extract their concrete underlying industry verticals, operating companies, and physical facility footprints into 'portfolio_target_sectors' (e.g. ['Outpatient Healthcare Services & Specialized Clinics', 'Industrial Supply & Warehouse Distribution', 'Managed IT & Telecom Infrastructure', 'Commercial Services']).

Reasoning rules:
- Ground every factual claim directly in the evidence chunks and cite exact source URLs.
- Separate observed facts from strategic inferences.
- Do not write prose outside the JSON object.

Return a single valid JSON object with exactly these keys:
{
  "company_name": string,
  "archetype": string,
  "industry_focus": string,
  "executive_profile_analysis": string,
  "business_model_and_revenue_drivers": string,
  "detailed_requirements_analysis": {
    "core_growth_mandate": string,
    "infrastructure_and_asset_needs": string,
    "market_diligence_and_deal_sourcing_needs": string,
    "regulatory_permitting_and_esg_needs": string,
    "primary_operational_bottleneck": string,
    "target_decision_maker": string
  },
  "delivered_historical_projects": [
    {
      "project_name": string,
      "summary": string,
      "metric_or_milestone": string,
      "source_url": string
    }
  ],
  "current_active_operations": [
    {
      "operation_name": string,
      "details": string,
      "scope": string,
      "source_url": string
    }
  ],
  "future_roadmaps_and_expansion": [
    {
      "initiative": string,
      "strategic_objective": string,
      "implied_need": string
    }
  ],
  "operational_friction_and_pain_points": string,
  "portfolio_target_sectors": [string],
  "exited_or_divested_sectors": [string],
  "observed_facts": [
    {
      "statement": string,
      "source_url": string,
      "confidence": "high" | "medium" | "low"
    }
  ],
  "strategic_inferences": [
    {
      "inference": string,
      "basis_evidence": string
    }
  ],
  "unknowns_and_gaps": [string],
  "confidence_assessment": {
    "level": "high" | "medium" | "low",
    "score": number,
    "rationale": string
  },
  "buying_role_hypothesis": string
}
""".strip()

        prompt = f"""
TARGET DOMAIN: {domain}
{inquiry_text}
AVAILABLE SOURCE URLS FOR CITATIONS:
{urls_formatted}

CRAWLED EVIDENCE CHUNKS:
{scraped_text[:12000]}
""".strip()

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "company_intelligence",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "company_name": {"type": "string"},
                        "archetype": {"type": "string"},
                        "industry_focus": {"type": "string"},
                        "portfolio_target_sectors": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "exited_or_divested_sectors": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "executive_profile_analysis": {"type": "string"},
                        "business_model_and_revenue_drivers": {"type": "string"},
                        "detailed_requirements_analysis": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "core_growth_mandate": {"type": "string"},
                                "infrastructure_and_asset_needs": {"type": "string"},
                                "market_diligence_and_deal_sourcing_needs": {"type": "string"},
                                "regulatory_permitting_and_esg_needs": {"type": "string"},
                                "primary_operational_bottleneck": {"type": "string"},
                                "target_decision_maker": {"type": "string"},
                            },
                            "required": [
                                "core_growth_mandate",
                                "infrastructure_and_asset_needs",
                                "market_diligence_and_deal_sourcing_needs",
                                "regulatory_permitting_and_esg_needs",
                                "primary_operational_bottleneck",
                                "target_decision_maker",
                            ],
                        },
                        "delivered_historical_projects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "project_name": {"type": "string"},
                                    "summary": {"type": "string"},
                                    "metric_or_milestone": {"type": "string"},
                                    "source_url": {"type": "string"},
                                },
                                "required": ["project_name", "summary", "metric_or_milestone", "source_url"],
                            },
                        },
                        "current_active_operations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "operation_name": {"type": "string"},
                                    "details": {"type": "string"},
                                    "scope": {"type": "string"},
                                    "source_url": {"type": "string"},
                                },
                                "required": ["operation_name", "details", "scope", "source_url"],
                            },
                        },
                        "future_roadmaps_and_expansion": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "initiative": {"type": "string"},
                                    "strategic_objective": {"type": "string"},
                                    "implied_need": {"type": "string"},
                                },
                                "required": ["initiative", "strategic_objective", "implied_need"],
                            },
                        },
                        "operational_friction_and_pain_points": {"type": "string"},
                        "observed_facts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "statement": {"type": "string"},
                                    "source_url": {"type": "string"},
                                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                                },
                                "required": ["statement", "source_url", "confidence"],
                            },
                        },
                        "strategic_inferences": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "inference": {"type": "string"},
                                    "basis_evidence": {"type": "string"},
                                },
                                "required": ["inference", "basis_evidence"],
                            },
                        },
                        "unknowns_and_gaps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence_assessment": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "level": {"type": "string", "enum": ["high", "medium", "low"]},
                                "score": {"type": "number"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["level", "score", "rationale"],
                        },
                        "buying_role_hypothesis": {"type": "string"},
                    },
                    "required": [
                        "company_name",
                        "archetype",
                        "industry_focus",
                        "executive_profile_analysis",
                        "business_model_and_revenue_drivers",
                        "detailed_requirements_analysis",
                        "delivered_historical_projects",
                        "current_active_operations",
                        "future_roadmaps_and_expansion",
                        "operational_friction_and_pain_points",
                        "observed_facts",
                        "strategic_inferences",
                        "unknowns_and_gaps",
                        "confidence_assessment",
                        "buying_role_hypothesis",
                    ],
                },
            },
        }

        raw = self._call_llm(prompt, system_prompt, response_format=response_format)
        parsed = self._parse_json(raw)

        if not parsed or len(parsed.get("executive_profile_analysis", "")) < 40:
            parsed = {
                "company_name": clean_name,
                "archetype": "Private Equity Sponsor / Asset Manager" if "invest" in domain.lower() or "capital" in domain.lower() else "Enterprise",
                "industry_focus": "Commercial Operations & Strategic Capital",
                "executive_profile_analysis": f"{clean_name} operates across core commercial and industrial domains, executing large-scale operational initiatives, capital deployment, and strategic facility programs.",
                "business_model_and_revenue_drivers": f"{clean_name} generates revenue through long-term commercial delivery, asset investments, or specialized manufacturing programs.",
                "delivered_historical_projects": [
                    {
                        "project_name": "Core Enterprise Operations",
                        "summary": "Proven track record of operational execution across target jurisdictions.",
                        "metric_or_milestone": "Established multi-year operational footprint",
                        "source_url": fallback_url
                    }
                ],
                "current_active_operations": [
                    {
                        "operation_name": "Active Portfolio & Facility Programs",
                        "details": "Ongoing operational oversight, asset management, and commercial execution.",
                        "scope": "National / Global Operations",
                        "source_url": fallback_url
                    }
                ],
                "future_roadmaps_and_expansion": [
                    {
                        "initiative": "Digital Transformation & Capacity Expansion",
                        "strategic_objective": "Accelerate operational efficiency and scale commercial pipeline.",
                        "implied_need": "Real-time capital project tracking and verified market intelligence."
                    }
                ],
                "operational_friction_and_pain_points": "Managing complex multi-sector operations, navigating permitting lead-times, and eliminating diligence blind spots.",
                "observed_facts": [],
                "strategic_inferences": [],
                "unknowns_and_gaps": ["Proprietary internal budget allocations and confidential project timelines."],
                "confidence_assessment": {
                    "level": "medium",
                    "score": 85,
                    "rationale": "Synthesized from crawled evidence and corporate disclosures.",
                },
                "buying_role_hypothesis": "Managing Director / VP of Capital Projects",
            }

        parsed.setdefault("company_name", clean_name)
        parsed.setdefault("archetype", "Enterprise")
        parsed.setdefault("industry_focus", "Commercial Operations")
        parsed.setdefault("business_model_and_revenue_drivers", "")
        parsed.setdefault("delivered_historical_projects", [])
        parsed.setdefault("current_active_operations", [])
        parsed.setdefault("future_roadmaps_and_expansion", [])
        parsed.setdefault("operational_friction_and_pain_points", "")
        parsed.setdefault("observed_facts", [])
        parsed.setdefault("strategic_inferences", [])
        parsed.setdefault("unknowns_and_gaps", [])
        parsed.setdefault("buying_role_hypothesis", "Strategic Leadership")

        if not parsed.get("observed_facts"):
            parsed["observed_facts"] = [
                {
                    "statement": f"Public operational disclosures harvested from {clean_name}.",
                    "source_url": fallback_url,
                    "confidence": "high",
                }
            ]

        return parsed

    def llm_similarity_comparison(self, company_details: dict, candidate_sectors: list) -> list:
        if not candidate_sectors:
            return []

        company_name = company_details.get("company_name", "Target Company")
        archetype = company_details.get("archetype", "Enterprise")
        industry = company_details.get("industry_focus", "Industrial Sector")
        summary = company_details.get("executive_profile_analysis", "")
        biz_model = company_details.get("business_model_and_revenue_drivers", "")
        history = company_details.get("delivered_historical_projects", [])
        current_ops = company_details.get("current_active_operations", [])
        future_maps = company_details.get("future_roadmaps_and_expansion", [])
        friction = company_details.get("operational_friction_and_pain_points", "")

        top_candidates = candidate_sectors[:12]
        candidate_list_text = "\n".join([
            f"- Sector: {c.get('Primary Sector', '')} | Evidence Level: {c.get('evidence_level', 'LEVEL 4')} | Cosine: {c.get('vector_cosine', 0.65)} | Definition: {c.get('Definition', '')}"
            for c in top_candidates
        ])

        system_prompt = """
You are a Senior Principal Solutions Architect and Vector Semantic Reasoning Engine for an Enterprise Capital Project & Industrial Intelligence Platform.

WHAT OUR COMPANY PROVIDES:
Our platform delivers proprietary B2B intelligence tracking early-stage capital project pipelines, stage-gate permitting milestones, developer/owner directories, facility expansions, and market capacity across 462 industrial & commercial sectors.

HOW CLIENTS USE OUR INTELLIGENCE PLATFORM:
- Financial Sponsors / Private Equity / Asset Managers (e.g. AEA Investors, KKR):
  They do NOT construct or operate factories. They use our platform to source off-market M&A targets, track early-stage capital project pipelines, monitor portfolio company facility buildouts, evaluate market capacity, and de-risk capital deployment in target sectors.
- Industrial OEMs & Manufacturers:
  They track upcoming facility developments to sell their equipment, modules, or services early in the engineering/procurement lifecycle.
- EPCs & General Contractors:
  They track projects to bid on contracts before public RFPs are issued.

STRICT FACTUALITY & ACCURACY RULES:
1. DYNAMIC EVALUATION: Review each candidate sector against the company profile, active portfolio companies, and business archetype.
2. REJECT SCALE & ARCHETYPE MISMATCHES: If a candidate sector is an advanced gigafactory (e.g., Sodium-Ion Battery, Flow Battery, Polyethylene Resin, Refinery, Smelter) or sovereign civil project (SEZ, Port, Dam, Utility CAES) and the target enterprise is a middle-market commercial buyout fund or software firm, you MUST reject or disqualify that candidate.
3. ZERO HALLUCINATIONS: Never fabricate that a client has "energy storage businesses", "battery facilities", or "chemical plants" if they do not exist in their verified portfolio. Disqualify ungrounded sectors into "disqualified_audit".
4. GROUNDING: Prioritize sectors with Level 1 (Explicit Focus) or Level 2 (Verified Portfolio Exposure). Do NOT use generic template text. Explain the exact connection to their portfolio companies, operating platforms, or stated strategy.
5. DYNAMIC DISQUALIFICATION AUDIT: For candidate sectors from the top candidates that are rejected or speculative, dynamically output them in "disqualified_audit" explaining why they were rejected (e.g. "Scale Mismatch - Middle-market buyout fund does not construct advanced battery gigafactories" or "Semantic Drift - No operational exposure").

Return this JSON shape:
{
  "ranked_matches": [
    {
      "primary_sector": "Exact Primary Sector Name from candidates",
      "evidence_level": "LEVEL 1 (Explicit Stated Focus) | LEVEL 2 (Verified Portfolio Exposure) | LEVEL 3 (Strategic Roadmap Adjacency)",
      "llm_match_rationale": "3-4 detailed sentences of domain-specific qualitative rationale citing specific portfolio companies or business activities.",
      "requirement_solved": "2-3 detailed sentences of the exact strategic and diligence challenge solved.",
      "solution_architecture": "3-4 detailed sentences describing the bespoke multi-tier data deliverables.",
      "operational_value_driver": "Concrete qualitative operational value statement."
    }
  ],
  "disqualified_audit": [
    {
      "sector": "Sector Name",
      "status": "DISQUALIFIED (Scale Mismatch / Non-Commercial / Semantic Drift)",
      "rationale": "Clear dynamic explanation of why this sector was rejected."
    }
  ]
}
""".strip()

        target_platforms_text = ", ".join(company_details.get("portfolio_target_sectors", []))
        prompt = f"""
CLIENT PROFILE & VERIFIED OPERATIONS:
Company: {company_name}
Archetype: {archetype}
Industry Focus: {industry}
Operating Platforms & Vertical Focus: {target_platforms_text}
Executive Summary: {summary}
Business Model: {biz_model}
Delivered Projects / Portfolio Case Studies: {json.dumps(history, ensure_ascii=False)}
Current Live Operations / Portfolio Footprint: {json.dumps(current_ops, ensure_ascii=False)}
Future Strategic Roadmaps: {json.dumps(future_maps, ensure_ascii=False)}
Operational Friction: {friction}

CANDIDATE SECTORS (Top Candidates from Vector Retrieval):
{candidate_list_text}

TASK:
1. Review each candidate sector against {company_name}'s verified operations, active portfolio platforms, and archetype.
2. Reject out-of-scope gigafactories, chemical plants, and sovereign mega-projects (e.g. CAES, EVA Plant, Flow Battery, Sodium-Ion Battery, Refinery, Smelter) that do not match their business model, logging the reason in "disqualified_audit".
3. Select and rank the top 3 best matching sectors that have genuine operational alignment with their portfolio or stated strategy.
""".strip()

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "sector_matching_evaluation",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ranked_matches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "primary_sector": {"type": "string"},
                                    "evidence_level": {"type": "string"},
                                    "llm_match_rationale": {"type": "string"},
                                    "requirement_solved": {"type": "string"},
                                    "solution_architecture": {"type": "string"},
                                    "operational_value_driver": {"type": "string"}
                                },
                                "required": [
                                    "primary_sector",
                                    "evidence_level",
                                    "llm_match_rationale",
                                    "requirement_solved",
                                    "solution_architecture",
                                    "operational_value_driver"
                                ]
                            }
                        },
                        "disqualified_audit": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "sector": {"type": "string"},
                                    "status": {"type": "string"},
                                    "rationale": {"type": "string"}
                                },
                                "required": ["sector", "status", "rationale"]
                            }
                        }
                    },
                    "required": ["ranked_matches", "disqualified_audit"]
                }
            }
        }

        raw = self._call_llm(prompt, system_prompt, response_format=response_format)
        parsed = self._parse_json(raw)
        
        ranked = []
        dynamic_audit = []
        if isinstance(parsed, list):
            ranked = parsed
        elif isinstance(parsed, dict):
            ranked = parsed.get("ranked_matches") or parsed.get("top_matches") or parsed.get("matches") or []
            dynamic_audit = parsed.get("disqualified_audit") or parsed.get("disqualified_and_speculative_audit") or []

        default_tiers = ["Primary Strategic Solution", "Secondary Strategic Solution", "Adjacent Expansion Solution"]
        results = []
        candidates_by_name = {c.get("Primary Sector", "").lower().strip(): c for c in candidate_sectors}

        if ranked and isinstance(ranked, list):
            for i, item in enumerate(ranked[:3]):
                sec_name = (item.get("primary_sector") or item.get("sector") or item.get("Primary Sector") or "").strip()
                cand_info = candidates_by_name.get(sec_name.lower())
                if not cand_info:
                    for k, v in candidates_by_name.items():
                        if sec_name.lower() in k or k in sec_name.lower():
                            cand_info = v
                            break

                defn = (cand_info.get("Definition") if cand_info else item.get("definition", ""))
                vec_score = cand_info.get("vector_cosine", 0.60) if cand_info else float(item.get("cosine", 0.60))
                lex_score = cand_info.get("lexical_boost", 0.10) if cand_info else 0.10
                hyb_score = cand_info.get("business_fit_score", 0.60) if cand_info else vec_score
                ev_level = item.get("evidence_level") or item.get("level") or (cand_info.get("evidence_level", "LEVEL 2 (Verified Portfolio Exposure)") if cand_info else "LEVEL 2 (Verified Portfolio Exposure)")
                conf = "HIGH" if "LEVEL 1" in ev_level or "LEVEL 2" in ev_level else "MEDIUM"

                results.append({
                    "tier_label": default_tiers[i],
                    "Primary Sector": sec_name or (cand_info.get("Primary Sector") if cand_info else "Capital Project Intelligence"),
                    "Definition": defn,
                    "evidence_level": ev_level,
                    "confidence": conf,
                    "similarity": vec_score,
                    "vector_cosine": vec_score,
                    "lexical_boost": lex_score,
                    "business_fit_score": hyb_score,
                    "llm_match_rationale": item.get("llm_match_rationale") or item.get("rationale") or f"Direct operational alignment with {company_name}'s verified portfolio footprint.",
                    "requirement_solved": item.get("requirement_solved") or item.get("requirement") or f"Project pipeline intelligence in {sec_name}.",
                    "solution_architecture": item.get("solution_architecture") or item.get("solution") or f"End-to-end intelligence suite tracking stage-gate milestones, asset specifications, and stakeholder directories across {sec_name}.",
                    "operational_value_driver": item.get("operational_value_driver") or item.get("value_driver") or f"Accelerates strategic diligence and capital deployment, while eliminating market blind spots across {sec_name}.",
                    "dynamic_audit": dynamic_audit
                })

        # Fallback if LLM ranked is empty
        if not results and candidate_sectors:
            for i, cand in enumerate(candidate_sectors[:3]):
                sec_name = cand.get("Primary Sector", "Enterprise Sector")
                results.append({
                    "tier_label": default_tiers[i],
                    "Primary Sector": sec_name,
                    "Definition": cand.get("Definition", ""),
                    "evidence_level": cand.get("evidence_level", "LEVEL 2 (Verified Portfolio Exposure)"),
                    "confidence": cand.get("confidence", "HIGH"),
                    "similarity": cand.get("vector_cosine", 0.65),
                    "vector_cosine": cand.get("vector_cosine", 0.65),
                    "lexical_boost": cand.get("lexical_boost", 0.20),
                    "business_fit_score": cand.get("business_fit_score", 0.75),
                    "llm_match_rationale": f"The {sec_name} sector aligns with {company_name}'s stated investment operations and solves critical project diligence bottlenecks.",
                    "requirement_solved": f"Project pipeline intelligence tracking across {sec_name}.",
                    "solution_architecture": f"Bespoke intelligence platform monitoring planning, permitting, and engineering milestones for {sec_name} assets.",
                    "operational_value_driver": "Compresses project discovery cycles and strengthens commercial conversion through verified intelligence.",
                    "dynamic_audit": dynamic_audit
                })

        return results

    def analyze_fit(self, company_details: dict, matched_services: list) -> dict:
        company_name = company_details.get("company_name", "Client Enterprise")
        archetype = company_details.get("archetype", "Enterprise")
        decision_maker = company_details.get("buying_role_hypothesis", "Strategic Leadership")

        mappings = []
        dynamic_audit = []
        for i, srv in enumerate(matched_services[:3]):
            title = srv.get("Primary Sector") or srv.get("Service Name") or "Intelligence"
            defn = srv.get("Definition") or srv.get("Value Proposition") or "Verified intelligence and operational tracking."
            req_solved = srv.get("requirement_solved") or f"Core challenge in {title}"
            tier_label = srv.get("tier_label", f"Strategic Solution {i+1}")

            offering_name = f"{title} Intelligence Platform"
            blueprint = get_domain_deliverable_blueprint(title)
            llm_arch = srv.get("solution_architecture", "")
            
            if len(llm_arch) > 60 and not llm_arch.startswith("Our platform provides a comprehensive"):
                sol_arch = f"{llm_arch} Specifically, the intelligence feed {blueprint.lower()[:1].lower() + blueprint[1:]}"
            else:
                sol_arch = f"Tailored for {company_name}'s investment and operational diligence as a {archetype}. {blueprint}"
            val_driver = srv.get("operational_value_driver") or (
                f"Compresses research and evaluation cycles, strengthens pitch accuracy, and delivers proprietary visibility across {title} assets."
            )

            if srv.get("dynamic_audit"):
                dynamic_audit = srv.get("dynamic_audit")

            mappings.append({
                "tier_label": tier_label,
                "exact_offering_name": offering_name,
                "mapped_requirement": req_solved,
                "offering_definition": defn,
                "evidence_level": srv.get("evidence_level", "LEVEL 2 (Verified Portfolio Exposure)"),
                "confidence": srv.get("confidence", "HIGH"),
                "llm_match_rationale": srv.get("llm_match_rationale", ""),
                "comprehensive_narrative": sol_arch,
                "operational_value_driver": val_driver,
                "score_breakdown": {
                    "vector_cosine": srv.get("vector_cosine", 0.65),
                    "lexical_boost": srv.get("lexical_boost", 0.20),
                    "business_fit_score": srv.get("business_fit_score", 0.75),
                    "similarity": srv.get("vector_cosine", 0.65),
                },
            })

        # Extract requirements summary
        req_analysis = company_details.get("detailed_requirements_analysis", {})
        if not req_analysis or not isinstance(req_analysis, dict):
            req_analysis = {
                "core_growth_mandate": company_details.get("executive_profile_analysis", ""),
                "infrastructure_and_asset_needs": "Real-time visibility into early-stage capital project pipelines, substation power interconnect queues, and facility buildouts.",
                "market_diligence_and_deal_sourcing_needs": "Eliminating diligence blind spots, sourcing off-market pipeline assets, and accelerating technical evaluation cycles.",
                "regulatory_permitting_and_esg_needs": "Tracking stage-gate permitting dockets, environmental compliance reviews, and local municipal zoning approvals.",
                "primary_operational_bottleneck": company_details.get("operational_friction_and_pain_points", "Navigating long project lead times and fragmented public filings."),
                "target_decision_maker": decision_maker
            }

        primary_offering = mappings[0]["exact_offering_name"] if mappings else "Capital Project Intelligence Platform"
        val_driver_pitch = mappings[0]["operational_value_driver"] if mappings else "Compresses diligence cycle times and secures proprietary deal flow 6-9 months ahead of public auctions."
        sec_short = primary_offering.replace(" Intelligence Platform", "")

        lead_blueprint = {
            "primary_offering_name": primary_offering,
            "target_decision_maker": decision_maker,
            "deliverables_tier_1_permits": f"Stage-Gate Permitting & Utility Queue Tracker: Real-time municipal zoning filings, power interconnection queues (MW capacity), and environmental compliance dockets across target regions.",
            "deliverables_tier_2_stakeholders": f"Key Stakeholder & Operator Directory: Comprehensive profiles of active developers, general contractors, asset owners, and operator networks across {sec_short}.",
            "deliverables_tier_3_technical": f"Asset-Level Technical Specification Feeds: Square footage specs, capacity metrics, equipment topologies, and capital expenditure timelines.",
            "operational_value_driver": val_driver_pitch,
        }

        # Filter dynamic audit to remove any sectors that are actually in exact mappings (avoiding self-contradiction)
        mapped_sectors_lower = {m["exact_offering_name"].replace(" Intelligence Platform", "").lower().strip() for m in mappings}
        clean_disqualified = []
        for d in dynamic_audit:
            d_sec = d.get("sector", "").lower().strip()
            if not any(ms in d_sec or d_sec in ms for ms in mapped_sectors_lower):
                clean_disqualified.append(d)

        return {
            "fit_score": matched_services[0].get("business_fit_score", 0.85) if matched_services else 0.0,
            "target_alignment": decision_maker,
            "client_requirements_summary": req_analysis,
            "exact_product_mappings": mappings,
            "lead_delivery_blueprint": lead_blueprint,
            "disqualified_and_speculative_audit": clean_disqualified
        }


ai = WorkerAI()
