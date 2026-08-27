import os
import json
import re
import time
import requests
from typing import List, Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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
        inquiry_text = f'\nClient Specific Inbound Inquiry / Requirement:\n"{client_inquiry}"\n' if client_inquiry else ""

        pages = getattr(evidence_store, "pages", None) if evidence_store else None
        available_urls = [p.url for p in pages] if pages else [f"https://{domain}" if domain else ""]
        available_urls = [u for u in available_urls if u]
        fallback_url = available_urls[0] if available_urls else (f"https://{domain}" if domain else "")

        urls_formatted = "\n".join([f"- {u}" for u in available_urls[:10]])

        system_prompt = """
You are a Senior Principal Corporate Intelligence Analyst and Evidence Verification Specialist.

Analyze the target enterprise strictly from the provided crawled evidence. Use only the supplied evidence chunks and source URLs. Do not invent clients, roadmap items, technologies, executives, budgets, partnerships, or business activities that are not explicitly supported.

Reasoning rules:
- Ground every factual claim in the evidence.
- Separate observed facts from strategic inferences.
- If a field cannot be verified, return null, an empty array, or a short unknown value instead of guessing.
- Prefer specificity over generic business language.
- Use only the provided source URLs for citations.
- Do not write prose outside the JSON object.

Return a single valid JSON object with exactly these keys:
{
  "company_name": string,
  "archetype": string,
  "industry_focus": string,
  "executive_profile_analysis": string,
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

Field guidance:
- company_name: Use the official company name if evident; otherwise infer conservatively from the domain.
- archetype: Choose a specific enterprise archetype such as SaaS Operator, Manufacturer, Utility Developer, Healthcare Provider, General Contractor, Private Equity Sponsor, Distributor, Logistics Operator, or Technology OEM.
- industry_focus: State the primary industry or operating domain in one short phrase.
- executive_profile_analysis: 3 to 5 sentences summarizing what the company does, based only on evidence.
- observed_facts: Only include claims explicitly supported by evidence.
- strategic_inferences: Include only cautious inferences that follow logically from the facts.
- unknowns_and_gaps: List important things you cannot verify from the public evidence.
- confidence_assessment.score: Use a 0 to 100 integer.
- buying_role_hypothesis: Guess the most likely decision-maker role only if there is enough evidence; otherwise use "Unknown".

Quality bar:
The output must be concise, specific, and evidence-backed. Avoid generic filler, marketing language, or unsupported strategic claims.
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
                        "executive_profile_analysis": {"type": "string"},
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
                "archetype": "Unknown",
                "industry_focus": "Unknown",
                "executive_profile_analysis": "Insufficient evidence was available to produce a reliable company profile.",
                "observed_facts": [],
                "strategic_inferences": [],
                "unknowns_and_gaps": ["Insufficient public evidence to verify company operations, offerings, or priorities."],
                "confidence_assessment": {
                    "level": "low",
                    "score": 25,
                    "rationale": "The model response was missing, incomplete, or not evidence-backed enough to trust.",
                },
                "buying_role_hypothesis": "Unknown",
            }

        parsed.setdefault("company_name", clean_name)
        parsed.setdefault("archetype", "Unknown")
        parsed.setdefault("industry_focus", "Unknown")
        parsed.setdefault("observed_facts", [])
        parsed.setdefault("strategic_inferences", [])
        parsed.setdefault("unknowns_and_gaps", [])
        parsed.setdefault("buying_role_hypothesis", "Unknown")

        if not parsed.get("observed_facts"):
            parsed["observed_facts"] = [
                {
                    "statement": f"Public evidence was limited for {clean_name}.",
                    "source_url": fallback_url,
                    "confidence": "low",
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
        needs = company_details.get("unknowns_and_gaps", [])
        friction = company_details.get("strategic_inferences", [])

        top_candidates = candidate_sectors[:5]
        candidate_list_text = "\n".join([
            f"- Candidate #{i+1}: {c.get('Primary Sector', '')} | Match: {c.get('match_pct', 95.0)}% | Definition: {c.get('Definition', '')}"
            for i, c in enumerate(top_candidates)
        ])

        system_prompt = """
You are a Senior Principal Solutions Architect and Vector Semantic Reasoning Engine.

You are given candidate catalog sectors that were pre-ranked by hybrid vector similarity for this company. Your task is to review the top candidate sectors and provide a concise rationale explaining why each candidate sector aligns with the company's verified operations and solves their requirements.

Rules:
- Use only the provided company profile and candidate definitions.
- Do not invent operational requirements.
- Do not overstate certainty.
- Return strictly valid JSON.
- Keep the rationale specific and practical.

Return this JSON shape:
{
  "ranked_matches": [
    {
      "primary_sector": "Exact Primary Sector Name from candidates",
      "llm_match_rationale": "2-sentence explanation of operational and commercial fit.",
      "requirement_solved": "Exact operational requirement or strategic challenge solved."
    }
  ]
}
""".strip()

        prompt = f"""
CLIENT PROFILE:
Company: {company_name}
Archetype: {archetype}
Industry Focus: {industry}
Executive Summary: {summary}
Known Gaps: {json.dumps(needs, ensure_ascii=False)}
Strategic Inferences: {json.dumps(friction, ensure_ascii=False)}

HYBRID RANKED CANDIDATE SECTORS:
{candidate_list_text}

Evaluate the top 3 candidate sectors and explain why they match.
""".strip()

        raw = self._call_llm(prompt, system_prompt)
        parsed = self._parse_json(raw)
        ranked = parsed.get("ranked_matches", [])

        default_tiers = ["Primary Strategic Solution", "Secondary Strategic Solution", "Adjacent Expansion Solution"]
        results = []
        llm_lookup = {item.get("primary_sector", "").lower().strip(): item for item in ranked if isinstance(item, dict)}

        for i, cand in enumerate(top_candidates[:3]):
            sec_name = cand.get("Primary Sector", "Unknown Sector")
            defn = cand.get("Definition", "")
            match_pct = cand.get("match_pct", 95.0 - i * 3.0)

            llm_item = llm_lookup.get(sec_name.lower().strip())
            if not llm_item:
                for k, v in llm_lookup.items():
                    if k in sec_name.lower() or sec_name.lower() in k:
                        llm_item = v
                        break

            rationale = (
                llm_item.get("llm_match_rationale")
                if llm_item
                else f"The {sec_name} sector aligns with the company's stated profile and likely addresses the highest-priority operational gap."
            )
            req_solved = (
                llm_item.get("requirement_solved")
                if llm_item
                else f"Core operational challenge addressed by {sec_name}."
            )

            results.append({
                "tier_label": default_tiers[i],
                "Primary Sector": sec_name,
                "Definition": defn,
                "similarity": cand.get("similarity", 0.90),
                "match_pct": match_pct,
                "vector_cosine": cand.get("vector_cosine", 0.65),
                "lexical_boost": cand.get("lexical_boost", 0.20),
                "hybrid_score": cand.get("hybrid_score", 0.85),
                "llm_match_rationale": rationale,
                "requirement_solved": req_solved,
            })

        return results

    def analyze_fit(self, company_details: dict, matched_services: list) -> dict:
        company_name = company_details.get("company_name", "Client Enterprise")
        archetype = company_details.get("archetype", "Unknown")
        decision_maker = company_details.get("buying_role_hypothesis", "Strategic Leadership")

        mappings = []
        for i, srv in enumerate(matched_services[:3]):
            title = srv.get("Primary Sector") or srv.get("Service Name") or "Intelligence"
            defn = srv.get("Definition") or srv.get("Value Proposition") or "Verified intelligence and operational tracking."
            req_solved = srv.get("requirement_solved") or f"Core challenge in {title}"
            tier_label = srv.get("tier_label", f"Strategic Solution {i+1}")

            offering_name = f"{title} Intelligence Platform"
            comprehensive_narrative = (
                f"In response to {company_name}'s operating profile as a {archetype}, the {title} Intelligence Platform provides verified visibility, structured evidence, and actionable context. "
                f"It helps teams prioritize opportunities, reduce uncertainty, and align commercial execution with observed business signals."
            )
            roi_narrative = (
                "Improves targeting accuracy, reduces manual research time, and strengthens pitch relevance by grounding recommendations in verified evidence."
            )

            mappings.append({
                "tier_label": tier_label,
                "exact_offering_name": offering_name,
                "mapped_requirement": req_solved,
                "offering_definition": defn,
                "llm_match_rationale": srv.get("llm_match_rationale", ""),
                "comprehensive_narrative": comprehensive_narrative,
                "roi_narrative": roi_narrative,
                "score_breakdown": {
                    "vector_cosine": srv.get("vector_cosine", 0.65),
                    "lexical_boost": srv.get("lexical_boost", 0.20),
                    "hybrid_score": srv.get("hybrid_score", 0.85),
                    "match_pct": srv.get("match_pct", 95.0),
                },
            })

        return {
            "fit_score": matched_services[0].get("match_pct", 98.0) if matched_services else 0.0,
            "target_alignment": decision_maker,
            "exact_product_mappings": mappings,
        }


ai = WorkerAI()
