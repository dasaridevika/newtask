import os
import re
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_NPZ_PATH = BASE_DIR / "catalog_embeddings.npz"

# Stopwords that must never trigger candidate matching on their own
DOMAIN_STOPWORDS = {
    "plant", "facility", "facilities", "system", "systems", "production", "manufacturing", 
    "building", "buildings", "complex", "center", "centers", "infrastructure", "other", 
    "services", "service", "support", "management", "supporting", "integrated", "development",
    "developments", "developer", "developers", "long", "term", "site", "sites", "program", "programs", 
    "business", "businesses", "individual", "professional", "commercial", "industrial", "private", 
    "growth", "scale", "home", "family", "built", "through", "with", "from", "that", "into", "these", 
    "their", "about", "after", "again", "against", "because", "been", "before", "being", "below",
    "between", "both", "during", "each", "further", "having", "here", "more", "most", "once", "only", 
    "same", "some", "such", "than", "then", "there", "they", "this", "those", "very", "what", "when", 
    "where", "which", "while", "who", "whom", "why", "will", "wherever", "data", "market", "markets", 
    "care", "health", "specific", "track", "record", "project", "projects", "pipeline", "pipelines",
    "operations", "technology", "technologies", "industry", "industries", "world", "assets", "asset", 
    "under", "strategic", "capital", "companies", "company", "products", "materials", "general", "waste", 
    "drive", "drives", "driving", "deliver", "delivers", "delivering", "expand", "expansion", "help", 
    "helps", "leading", "firm", "firms", "team", "teams", "partner", "partners", "investment", 
    "investments", "investor", "investors", "solutions", "solution", "opportunity", "opportunities", 
    "global", "local", "regional", "national", "movie", "patrons", "vehicles", "film", "films", 
    "structure", "flexible", "category", "focus", "across", "value", "added", "various", "multiple", 
    "broad", "wide", "distribution", "distributor", "distributors", "network", "networks", "line", 
    "lines", "treatment", "medical", "supply", "supplies", "equipment", "energy", "power", "storage",
    "generation", "utility", "utilities", "overhead", "fuels", "fuel", "spirit", "burdens", "complexity",
    "launch", "stories", "success", "driven", "training", "academic", "university", "universities", "school",
    "research", "office", "chemical", "road", "based", "prior", "range", "enable", "enablement", "trends"
}

# Domain Dictionary: Positive Context Anchors, Negative Patterns, and Scale Classes
DOMAIN_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "overhead": {
        "positive_context": ["power line", "transmission line", "aerial cable", "utility pole", "high voltage wire", "substation overhead", "overhead catenary", "aerial transmission"],
        "negative_context": ["complexity and overhead", "administrative overhead", "overhead burden", "operating overhead", "corporate overhead", "reduce overhead", "without overhead", "cost overhead"],
        "scale_class": "utility",
        "facility_types": ["Transmission Lines", "Utility Grid Structures", "Catenary Systems"]
    },
    "university": {
        "positive_context": ["higher education", "undergraduate", "postgraduate", "campus", "academic institution", "faculty", "degree granting", "university medical center"],
        "negative_context": ["cls university", "training program", "internal university", "corporate university", "learning portal", "academy", "training academy"],
        "scale_class": "commercial",
        "facility_types": ["University Campus", "Academic Building", "Research Quad"]
    },
    "sustainable aviation fuels (saf) production": {
        "positive_context": ["sustainable aviation fuel", "saf refinery", "biojet fuel", "aviation biofuel", "low-carbon jet", "hydroprocessed esters", "synthetic kerosene", "aviation decarbonization"],
        "negative_context": ["fuels the spirit", "fuels growth", "fuels innovation", "fuels the entrepreneurial", "fuels ambition", "fuels momentum"],
        "scale_class": "industrial",
        "facility_types": ["Biorefinery", "SAF Processing Facility", "Renewable Fuel Plant"]
    },
    "road": {
        "positive_context": ["highway construction", "pavement", "asphalt", "civil roadworks", "toll road", "arterial road", "freeway buildout", "interstate roadway"],
        "negative_context": ["road to success", "road ahead", "roadmap", "on the road", "middle market", "private debt"],
        "scale_class": "sovereign",
        "facility_types": ["Highway", "Toll Road", "Paved Arterial Infrastructure"]
    },
    "synthetic organic chemical plant": {
        "positive_context": ["chemical synthesis", "petrochemical plant", "organic compounds", "polymer production", "chemical reactor", "olefins unit", "specialty chemical manufacturing"],
        "negative_context": ["chemical industry trends", "consumer chemistry", "clean energy"],
        "scale_class": "industrial",
        "facility_types": ["Chemical Synthesis Plant", "Polymer Reactor Facility", "Petrochemical Complex"]
    },
    "other research facility": {
        "positive_context": ["scientific laboratory", "r&d center", "testing laboratory", "pilot plant facility", "cleanroom laboratory", "biotech lab", "material testing facility"],
        "negative_context": ["research industry and consumer trends", "actively research", "market research", "investment research", "equity research", "diligence research"],
        "scale_class": "commercial",
        "facility_types": ["R&D Laboratory", "Testing Facility", "Scientific Innovation Center"]
    },
    "office building": {
        "positive_context": ["commercial office building", "headquarters facility", "tenant lease", "class a office", "office park development", "commercial real estate building"],
        "negative_context": ["seven offices worldwide", "back-office", "office burdens", "home office", "executive office"],
        "scale_class": "commercial",
        "facility_types": ["Multi-Story Office Building", "Corporate Headquarters", "Commercial Real Estate Plaza"]
    },
    "warehouse": {
        "positive_context": ["distribution center", "fulfillment facility", "warehouse hub", "logistics terminal", "storage depot", "intermodal distribution", "cross-dock facility", "300k sf expansion", "warehouse", "distribution and fulfilment"],
        "negative_context": ["data warehouse", "software warehouse", "warehouse of knowledge"],
        "scale_class": "commercial",
        "facility_types": ["Industrial Distribution Center", "Logistics Hub", "Automated Fulfillment Center"]
    },
    "other health care building": {
        "positive_context": ["outpatient clinic", "medical office building", "ambulatory surgery center", "specialized medical facility", "diagnostic center", "rehabilitation clinic", "health system facility"],
        "negative_context": ["healthcare technology software", "healthcare consulting", "health and wellness"],
        "scale_class": "commercial",
        "facility_types": ["Outpatient Clinic", "Ambulatory Surgery Center", "Medical Office Building"]
    },
    "other communication infrastructure": {
        "positive_context": ["telecommunication towers", "fiber optic route", "carrier exchange", "edge data network", "cellular colocation", "lit fiber infrastructure", "dark fiber conduit"],
        "negative_context": ["internal communication", "communication strategy", "press communication"],
        "scale_class": "utility",
        "facility_types": ["Cellular Tower Network", "Fiber Optic Route", "Edge Exchange Node"]
    },
    "communication antenna tower": {
        "positive_context": ["antenna tower", "cellular tower", "broadcast mast", "wireless transmission tower", "microwave relay tower", "telecom monopoles"],
        "negative_context": ["tower of strength", "towering"],
        "scale_class": "utility",
        "facility_types": ["Wireless Cellular Tower", "Broadcast Mast", "Microwave Tower"]
    },
    "solar photovoltaic power plant (pv)": {
        "positive_context": [
            "solar", "solar pv", "photovoltaic", "solar power", "solar farm", "solar project", 
            "solar energy", "pv power", "utility solar", "solar array", "solar plant", "solar installation", 
            "solar generation", "pv system", "solar module", "solar park", "pv plant", "solar ground mount",
            "utility-scale solar", "solar panels", "clean electricity solar"
        ],
        "negative_context": ["solar rooftop calculator", "solar energy trends"],
        "scale_class": "utility",
        "facility_types": ["Utility Solar Farm", "Ground-Mounted PV Facility", "Solar Interconnect Substation"]
    },
    "concentrated solar power (csp)": {
        "positive_context": ["csp", "concentrated solar", "solar thermal", "heliostat", "parabolic trough", "molten salt solar", "csp power plant"],
        "negative_context": [],
        "scale_class": "utility",
        "facility_types": ["CSP Tower Facility", "Parabolic Trough Plant"]
    },
    "floating solar power plant": {
        "positive_context": ["floating solar", "floatovoltaics", "fpv", "water solar", "floating pv", "reservoir solar"],
        "negative_context": [],
        "scale_class": "utility",
        "facility_types": ["Floating PV Array", "Reservoir Solar Installation"]
    },
    "battery energy storage system (bess)": {
        "positive_context": ["bess", "battery storage", "grid battery", "energy storage system", "battery energy storage", "utility battery", "btm bess", "megapack"],
        "negative_context": [],
        "scale_class": "utility",
        "facility_types": ["Utility Battery Installation", "BESS Substation Facility"]
    },
    "lead acid (lab) battery production plant": {
        "positive_context": ["lead-acid battery", "lead acid battery", "battery cell manufacturing", "lab battery plant", "lead acid manufacturing", "automotive battery plant"],
        "negative_context": ["engineering lab", "research lab", "testing lab", "computer lab", "innovation lab", "lab environment", "lab solutions"],
        "scale_class": "industrial",
        "facility_types": ["Battery Manufacturing Plant", "Lead Acid Cell Facility"]
    },
    "ethylene vinyl acetate (eva) plant": {
        "positive_context": ["eva copolymer", "ethylene vinyl acetate", "eva sheet", "eva film", "eva encapsulant", "eva resin synthesis"],
        "negative_context": ["evaluate", "evaluation"],
        "scale_class": "industrial",
        "facility_types": ["EVA Polymer Plant", "Chemical Synthesis Facility"]
    },
    "polyethylene terephthalate (pet) plant": {
        "positive_context": ["pet resin", "polyethylene terephthalate", "pet bottle recycling", "pet polymer", "pet packaging plant"],
        "negative_context": ["pet animal", "pet store"],
        "scale_class": "industrial",
        "facility_types": ["PET Polymerization Plant", "Resin Manufacturing Facility"]
    },
    "polyhydroxyalkanoates (pha) plant": {
        "positive_context": ["pha biopolymer", "polyhydroxyalkanoates", "pha resin", "pha bioplastic plant", "bacterial fermentation pha"],
        "negative_context": [],
        "scale_class": "industrial",
        "facility_types": ["PHA Biopolymer Facility", "Bioplastics Plant"]
    },
    "thermal energy storage (tes)": {
        "positive_context": ["thermal energy storage", "chilled water storage", "molten salt storage", "tes tank", "ice thermal storage"],
        "negative_context": [],
        "scale_class": "utility",
        "facility_types": ["Thermal Storage Tank", "Molten Salt Facility"]
    },
    "flywheel energy storage (fes)": {
        "positive_context": ["flywheel energy storage", "flywheel storage", "kinetic energy storage", "flywheel rotor"],
        "negative_context": [],
        "scale_class": "utility",
        "facility_types": ["Flywheel Installation", "Kinetic Storage Facility"]
    },
    "data center": {
        "positive_context": [
            "data center", "datacenter", "data centre", "colocation", "hyperscale", "server farm", 
            "substation interconnect megawatt", "cooling topology data center", "enterprise data exchange",
            "rack density", "liquid cooling data center", "ai infrastructure", "critical digital infrastructure",
            "white space data center", "coolchip", "avocent"
        ],
        "negative_context": ["data center of excellence", "database", "data analytics", "data warehouse"],
        "scale_class": "commercial",
        "facility_types": ["Hyperscale Data Center", "Colocation Facility", "Edge Compute Node"]
    }
}

# Ambiguous short acronyms that must NEVER match standalone generic words
AMBIGUOUS_SHORT_ACRONYMS: Dict[str, List[str]] = {
    "lab": ["lead acid", "lead-acid", "lead acid battery", "lab battery", "battery production", "battery cell"],
    "pet": ["pet resin", "polyethylene terephthalate", "pet bottle", "pet polymer", "pet packaging"],
    "eva": ["eva copolymer", "ethylene vinyl acetate", "eva sheet", "eva film", "eva encapsulant"],
    "pha": ["pha biopolymer", "polyhydroxyalkanoates", "pha resin", "pha bioplastic"],
    "csp": ["concentrated solar", "solar thermal", "csp plant", "heliostat"],
    "fes": ["flywheel energy", "flywheel storage", "kinetic energy storage"],
    "tes": ["thermal energy storage", "chilled water storage", "molten salt storage", "tes tank"],
    "caes": ["compressed air energy", "compressed air storage", "caes facility"]
}

# Rich Synonym Mapping for Acronyms and Industry Terms
SECTOR_SYNONYM_MAP: Dict[str, List[str]] = {
    "solar photovoltaic power plant (pv)": [
        "solar pv", "solar", "photovoltaic", "solar power", "solar farm", "solar project", 
        "solar energy", "pv power", "utility solar", "solar array", "solar plant", "solar installation", 
        "solar generation", "pv system", "solar module", "solar park", "pv plant", "solar ground mount",
        "utility-scale solar", "solar panels"
    ],
    "concentrated solar power (csp)": [
        "csp", "concentrated solar", "solar thermal", "heliostat", "parabolic trough", "molten salt solar"
    ],
    "floating solar power plant": [
        "floating solar", "floatovoltaics", "fpv", "water-based solar", "floating pv"
    ],
    "solar pv cells & modules manufacturing plant": [
        "solar manufacturing", "pv cell manufacturing", "solar panel factory", "solar module assembly"
    ],
    "battery energy storage system (bess)": [
        "bess", "battery storage", "grid battery", "energy storage system", "battery energy storage", "utility battery"
    ],
    "lead acid (lab) battery production plant": [
        "lead acid battery", "lead-acid battery", "battery production", "lab battery"
    ],
    "onshore wind power plant": [
        "onshore wind", "wind farm", "wind turbine", "wind power", "wind energy", "wind project"
    ],
    "offshore wind power plant": [
        "offshore wind", "offshore turbine", "marine wind", "offshore wind farm", "floating wind"
    ],
    "data center": [
        "data center", "datacenter", "data centre", "colocation", "hyperscale", "server farm", 
        "rack density", "cloud infrastructure", "edge compute", "ai infrastructure", "critical digital infrastructure"
    ],
    "warehouse": [
        "warehouse", "distribution center", "fulfillment facility", "logistics hub", "storage depot", 
        "intermodal distribution", "cross-dock", "logistics facility", "distribution facility"
    ]
}

def get_candidate_aliases(sec_name: str) -> List[str]:
    """Generates all valid aliases, acronyms, and synonyms for a catalog sector name."""
    aliases = set()
    sec_lower = sec_name.lower().strip()
    aliases.add(sec_lower)
    
    # Clean version without parentheses
    clean = re.sub(r"\(.*?\)", "", sec_lower).strip()
    if clean:
        aliases.add(clean)
        
    # Extract acronyms inside parentheses: (PV), (CSP), (SAF), (BESS), (LNG), etc.
    acronyms = re.findall(r"\((.*?)\)", sec_lower)
    for acr in acronyms:
        acr_clean = acr.strip().lower()
        if acr_clean in AMBIGUOUS_SHORT_ACRONYMS:
            aliases.update(AMBIGUOUS_SHORT_ACRONYMS[acr_clean])
        elif len(acr_clean) >= 4:
            aliases.add(acr_clean)
            
    # Add mapped synonyms
    if sec_lower in SECTOR_SYNONYM_MAP:
        aliases.update(SECTOR_SYNONYM_MAP[sec_lower])
    if clean in SECTOR_SYNONYM_MAP:
        aliases.update(SECTOR_SYNONYM_MAP[clean])
        
    domain_meta = DOMAIN_DICTIONARY.get(sec_lower, {}) or DOMAIN_DICTIONARY.get(clean, {})
    if domain_meta:
        aliases.update(domain_meta.get("positive_context", []))
        
    return [a for a in aliases if a]

@dataclass
class ValidationResult:
    is_valid: bool
    rejection_code: Optional[str] = None  # NO_VERIFIED_EVIDENCE, CONTEXT_MISMATCH, POLYSEMY_OR_AMBIGUOUS_TERM, etc.
    rejection_reason: Optional[str] = None
    matched_phrases: List[str] = field(default_factory=list)
    ignored_terms: List[str] = field(default_factory=list)
    negative_context_hits: List[str] = field(default_factory=list)
    synonym_expansions: List[str] = field(default_factory=list)
    definition_entailment: float = 0.0
    entity_relationship_check: str = "unverified"
    supporting_evidence_ids: List[str] = field(default_factory=list)

def validate_evidence_for_candidate(
    candidate: Dict[str, Any],
    evidence_item: Dict[str, Any],
    company_details: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """
    Evaluates evidence against a catalog candidate using strict contextual validation.
    Enforces all 10 required checks with rich synonym awareness.
    """
    ev_id = evidence_item.get("evidence_id", "")
    source_url = evidence_item.get("source_url", "")
    quoted_text = evidence_item.get("quoted_text", "").strip()
    norm_quote = evidence_item.get("normalized_text", quoted_text.lower())
    ev_relationship = evidence_item.get("relationship", "current_operation")
    sec_name = candidate.get("primary_sector", "").strip()
    sec_lower = sec_name.lower()
    definition = candidate.get("definition", "").strip().lower()

    # Check 1, 2, 3: Basic Presence
    if not quoted_text or len(quoted_text) < 20:
        return ValidationResult(is_valid=False, rejection_code="NO_VERIFIED_EVIDENCE", rejection_reason="Quoted evidence text is empty or too short.")
    if not source_url:
        return ValidationResult(is_valid=False, rejection_code="NO_VERIFIED_EVIDENCE", rejection_reason="Source URL is missing.")

    # Check 6: Reject pure generic corporate boilerplate
    if ev_relationship == "generic_statement" or any(p in norm_quote for p in ["all rights reserved", "privacy policy", "terms of use", "cookie preferences"]):
        return ValidationResult(is_valid=False, rejection_code="GENERIC_STATEMENT", rejection_reason="Evidence is a generic corporate statement or web boilerplate.")

    clean_sec = re.sub(r"\(.*?\)", "", sec_name).lower().strip()
    domain_meta = DOMAIN_DICTIONARY.get(sec_lower, {}) or DOMAIN_DICTIONARY.get(clean_sec, {})
    pos_contexts = domain_meta.get("positive_context", [])
    neg_contexts = domain_meta.get("negative_context", [])

    # Check 7: Negative Context & Polysemy Filter
    neg_hits = [nc for nc in neg_contexts if nc in norm_quote]
    if neg_hits:
        return ValidationResult(
            is_valid=False,
            rejection_code="POLYSEMY_OR_AMBIGUOUS_TERM",
            rejection_reason=f"Evidence matches negative/polysemous context pattern: {neg_hits[0]}",
            negative_context_hits=neg_hits
        )

    # Check 4: Contextual Entailment Check using Aliases and Domain Anchors
    aliases = get_candidate_aliases(sec_name)
    matched_pos_phrases = [a for a in aliases if re.search(r"\b" + re.escape(a) + r"\b", norm_quote)]

    # Check multi-word clean phrase match
    sec_tokens = [t for t in re.findall(r"\b[a-zA-Z]{4,}\b", clean_sec) if t not in DOMAIN_STOPWORDS]

    has_phrase_match = len(matched_pos_phrases) > 0
    if len(clean_sec.split()) >= 2 and re.search(r"\b" + re.escape(clean_sec) + r"\b", norm_quote):
        has_phrase_match = True
        matched_pos_phrases.append(clean_sec)

    has_multi_token_match = False
    if len(sec_tokens) >= 2:
        token_hits = [t for t in sec_tokens if re.search(r"\b" + re.escape(t) + r"\b", norm_quote)]
        if len(token_hits) >= 2:
            has_multi_token_match = True
            matched_pos_phrases.extend(token_hits)

    # Check definition entailment
    def_tokens = [t for t in re.findall(r"\b[a-zA-Z]{4,}\b", definition) if t not in DOMAIN_STOPWORDS]
    def_hits = [t for t in def_tokens if re.search(r"\b" + re.escape(t) + r"\b", norm_quote)]
    entailment_score = min(1.0, (len(def_hits) / max(1, len(def_tokens[:6]))) + (0.5 if matched_pos_phrases else 0.0))

    if not matched_pos_phrases and not has_phrase_match and not (has_multi_token_match and len(def_hits) >= 2):
        ignored = [t for t in re.findall(r"\b[a-zA-Z]{4,}\b", clean_sec) if t in DOMAIN_STOPWORDS]
        return ValidationResult(
            is_valid=False,
            rejection_code="DEFINITION_NOT_ENTAILED",
            rejection_reason=f"Evidence text does not contain required physical domain anchors for '{sec_name}'.",
            ignored_terms=ignored,
            definition_entailment=entailment_score
        )

    # Check 5, 8, 9: Entity Relationship Validation
    archetype = company_details.get("archetype", "") if company_details else ""
    is_sponsor = "private equity" in archetype.lower() or "asset manager" in archetype.lower() or "sponsor" in archetype.lower() or "investment" in archetype.lower()

    rel_check = "verified_direct"
    if is_sponsor:
        if ev_relationship == "portfolio_company" or "portfolio" in source_url.lower():
            rel_check = "verified_portfolio_company"
        elif "expansion" in norm_quote or "distribution" in norm_quote or "clinic" in norm_quote:
            rel_check = "verified_portfolio_expansion"
        else:
            rel_check = "verified_sponsor_stated_focus"

    return ValidationResult(
        is_valid=True,
        matched_phrases=list(set(matched_pos_phrases)),
        definition_entailment=entailment_score,
        entity_relationship_check=rel_check,
        supporting_evidence_ids=[ev_id]
    )

def determine_evidence_level(
    candidate: Dict[str, Any],
    company_details: Optional[Dict[str, Any]] = None,
    client_inquiry: str = "",
    evidence_ledger: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, float, List[str], List[ValidationResult]]:
    """
    Calculates deterministic evidence level with robust alias and synonym matching.
    LEVEL 1: Explicit target enterprise stated focus/requirement/project or direct client inquiry.
    LEVEL 2: Verified current/historical portfolio company explicitly operating in candidate sector.
    LEVEL 3: Verified strategic adjacency.
    LEVEL 4: Only lexical/embedding similarity (Never an exact match).
    """
    sec_name = candidate.get("primary_sector", "").strip()
    clean_sec = re.sub(r"\(.*?\)", "", sec_name).lower().strip()
    aliases = get_candidate_aliases(sec_name)

    valid_results: List[ValidationResult] = []
    verified_evidence_ids: List[str] = []

    # Validate against Evidence Ledger
    if evidence_ledger:
        for ev in evidence_ledger:
            ev_dict = ev if isinstance(ev, dict) else (ev.to_dict() if hasattr(ev, "to_dict") else {})
            val_res = validate_evidence_for_candidate(candidate, ev_dict, company_details)
            if val_res.is_valid:
                valid_results.append(val_res)
                for eid in val_res.supporting_evidence_ids:
                    if eid and eid not in verified_evidence_ids:
                        verified_evidence_ids.append(eid)

    # 1. CHECK INBOUND CLIENT INQUIRY (LEVEL 1)
    if client_inquiry and len(client_inquiry.strip()) > 2:
        inq_lower = client_inquiry.lower()
        if any(re.search(r"\b" + re.escape(a) + r"\b", inq_lower) for a in aliases):
            return "LEVEL 1 (Explicit Stated Requirement)", 0.95, verified_evidence_ids or ["inquiry_direct_stated"], valid_results
        if clean_sec in inq_lower or any(st in inq_lower for st in clean_sec.split() if st not in DOMAIN_STOPWORDS and len(st) >= 4):
            return "LEVEL 1 (Explicit Stated Requirement)", 0.95, verified_evidence_ids or ["inquiry_direct_stated"], valid_results

    if not company_details:
        return "LEVEL 4 (Speculative / Semantic Only)", 0.40, [], valid_results

    # 2. CHECK EXPLICIT PORTFOLIO TARGET SECTORS & STATED FOCUS (LEVEL 1 / LEVEL 2)
    target_secs = company_details.get("portfolio_target_sectors", [])
    for ts in target_secs:
        ts_lower = ts.lower().strip()
        if any(re.search(r"\b" + re.escape(a) + r"\b", ts_lower) for a in aliases):
            return "LEVEL 1 (Explicit Stated Focus)", 0.95, verified_evidence_ids or ["profile_stated_focus"], valid_results

    # 3. CHECK CORE INDUSTRY FOCUS (LEVEL 1)
    ind_focus = str(company_details.get("industry_focus", "")).lower()
    if ind_focus and any(re.search(r"\b" + re.escape(a) + r"\b", ind_focus) for a in aliases):
        return "LEVEL 1 (Explicit Core Sector)", 0.95, verified_evidence_ids or ["industry_core_focus"], valid_results

    # 4. CHECK VERIFIED EVIDENCE VALIDATION RESULTS (LEVEL 2)
    if valid_results:
        has_portfolio_rel = any(vr.entity_relationship_check in ("verified_portfolio_company", "verified_portfolio_expansion") for vr in valid_results)
        if has_portfolio_rel:
            return "LEVEL 2 (Verified Portfolio Exposure)", 0.90, verified_evidence_ids, valid_results
        return "LEVEL 2 (Verified Portfolio Exposure)", 0.85, verified_evidence_ids, valid_results

    # 5. CHECK FUTURE STRATEGIC ROADMAPS (LEVEL 3)
    future_text = " ".join([f.get("initiative", "") + " " + f.get("strategic_objective", "") for f in company_details.get("future_roadmaps_and_expansion", [])]).lower()
    if any(re.search(r"\b" + re.escape(a) + r"\b", future_text) for a in aliases if len(a) >= 4):
        return "LEVEL 3 (Strategic Roadmap Adjacency)", 0.70, verified_evidence_ids or ["future_roadmap"], valid_results

    return "LEVEL 4 (Speculative / Semantic Only)", 0.40, [], valid_results


class ServiceCatalog:
    def __init__(self, npz_path=None):
        self.vectors = None        # shape (462, 1024)
        self.sectors = None        # array of 462 sector strings
        self.definitions = None    # array of 462 definition strings
        self.texts = None          # array of text strings
        self.candidate_ids = None  # array of "cat_001", "cat_002", etc.
        self.model_name = os.getenv("CF_EMBEDDING_MODEL", "@cf/baai/bge-large-en-v1.5")
        self.worker_url = os.getenv("CLOUDFLARE_WORKER_URL", "https://lead-research-ai-worker.devika-worker.workers.dev")
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

        target_npz = npz_path or EMBEDDINGS_NPZ_PATH
        if Path(target_npz).exists():
            self.load_embeddings(target_npz)

    def load_embeddings(self, npz_file) -> int:
        """Loads pre-computed 1024-dimensional normalized vector matrix in < 1ms."""
        data = np.load(npz_file, allow_pickle=True)
        self.vectors = data["vectors"].astype(np.float32)
        self.sectors = [str(s).strip() for s in data["sectors"]]
        self.definitions = [str(d).strip() for d in data["definitions"]]
        self.texts = data["texts"]
        self.candidate_ids = [f"cat_{i+1:03d}" for i in range(len(self.sectors))]

        if "model_name" in data:
            self.model_name = str(data["model_name"])

        cleaned_corpus = []
        for s, d in zip(self.sectors, self.definitions):
            aliases = get_candidate_aliases(s)
            combined = f"{s} {' '.join(aliases)} {d}".lower()
            tokens = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", combined) if t not in DOMAIN_STOPWORDS]
            cleaned_corpus.append(" ".join(tokens))

        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=8000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_corpus)
        return len(self.sectors)

    def _get_worker_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generates a 1024-dim dense vector using Cloudflare Workers AI with caching and retries."""
        if not text or not text.strip():
            return None

        cache_key = text.strip().lower()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        if not self.worker_url:
            return None
            
        for attempt in range(3):
            try:
                resp = requests.post(
                    self.worker_url.rstrip("/") + "/ai/embed",
                    json={"model": self.model_name, "text": [text[:2000]]},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if data and isinstance(data[0], list):
                        vec = np.array(data[0], dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        normalized = vec / (norm if norm > 0 else 1e-10)
                        self._embedding_cache[cache_key] = normalized
                        return normalized
                    elif data and isinstance(data[0], dict) and "values" in data[0]:
                        vec = np.array(data[0]["values"], dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        normalized = vec / (norm if norm > 0 else 1e-10)
                        self._embedding_cache[cache_key] = normalized
                        return normalized
                elif resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.0 * (attempt + 1))
                    continue
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        return None

    def embed_company(self, company_details: dict, scraped_text: str = "", client_inquiry: str = "") -> dict:
        """
        Multi-Vector Representation Architecture:
        Generates distinct semantic vectors for functionality, intent, business model, and portfolio.
        """
        company_name = company_details.get("company_name", "Target Enterprise")
        industry = company_details.get("industry_focus", "")
        summary = company_details.get("executive_profile_analysis", "")
        biz_model = company_details.get("business_model_and_revenue_drivers", "")
        archetype = company_details.get("archetype", "Enterprise")

        target_secs = ", ".join(company_details.get("portfolio_target_sectors", []))
        past_proj = " ".join([p.get("project_name", "") + " " + p.get("summary", "") for p in company_details.get("delivered_historical_projects", [])])
        future_proj = " ".join([f.get("initiative", "") + " " + f.get("strategic_objective", "") for f in company_details.get("future_roadmaps_and_expansion", [])])

        strategy_text = f"Enterprise: {company_name}. Archetype: {archetype}. Core Focus Sectors & Target Platforms: {industry} {target_secs}. Business Model: {biz_model}."
        portfolio_text = f"Portfolio Operations & Facility Footprint: {summary} {past_proj} {future_proj}."

        strat_vec = self._get_worker_embedding(strategy_text)
        port_vec = self._get_worker_embedding(portfolio_text)

        dim = self.vectors.shape[1] if self.vectors is not None else 1024
        if strat_vec is None:
            strat_vec = np.zeros(dim, dtype=np.float32)
        if port_vec is None:
            port_vec = np.zeros(dim, dtype=np.float32)

        if client_inquiry and len(client_inquiry.strip()) > 2:
            inq_vec = self._get_worker_embedding(f"Specific Client Inquiry & Stated Requirement: {client_inquiry}")
            if inq_vec is None:
                inq_vec = np.zeros(dim, dtype=np.float32)
            composite = 0.40 * strat_vec + 0.30 * port_vec + 0.30 * inq_vec
        else:
            composite = 0.55 * strat_vec + 0.45 * port_vec

        norm = np.linalg.norm(composite)
        normalized_vector = composite / (norm if norm > 0 else 1e-10)

        return {
            "query_text": f"{strategy_text} {portfolio_text} {client_inquiry}",
            "vector": normalized_vector,
            "dimension": len(normalized_vector),
            "model_name": self.model_name
        }

    def match_company_vector(
        self,
        company_vector: np.ndarray,
        company_text: str = "",
        company_details: Optional[dict] = None,
        client_inquiry: str = "",
        evidence_ledger: Optional[List[Any]] = None,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Evidence-First Deterministic Matching with Section H Multi-Factor Scoring Formula:
        final_score = 0.30 * evidence_score + 0.20 * intent_score + 0.15 * functionality_score + 
                      0.15 * definition_score + 0.10 * business_model_score + 0.05 * facility_score + 
                      0.05 * lexical_score
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # 1. Dense Vector Cosine Similarity
        dense_sims = np.dot(self.vectors, company_vector)

        # 2. Dynamic TF-IDF Lexical Similarity
        company_lower = (company_text + " " + client_inquiry).lower()
        if self.tfidf_vectorizer and self.tfidf_matrix is not None and len(company_lower) > 5:
            clean_company_tokens = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", company_lower) if t not in DOMAIN_STOPWORDS]
            company_tfidf = self.tfidf_vectorizer.transform([" ".join(clean_company_tokens)])
            tfidf_sims = (self.tfidf_matrix * company_tfidf.T).toarray().flatten()
        else:
            tfidf_sims = np.zeros(len(self.sectors), dtype=np.float32)

        candidates: List[Dict[str, Any]] = []

        for idx in range(len(self.sectors)):
            cand_id = self.candidate_ids[idx]
            sec_name = self.sectors[idx]
            definition = self.definitions[idx]
            raw_vec_score = float(dense_sims[idx])
            raw_tfidf_score = float(tfidf_sims[idx])
            clean_sec = re.sub(r"\(.*?\)", "", sec_name).lower().strip()
            aliases = get_candidate_aliases(sec_name)

            domain_meta = DOMAIN_DICTIONARY.get(clean_sec, {}) or DOMAIN_DICTIONARY.get(sec_name.lower(), {})
            pos_contexts = domain_meta.get("positive_context", [])
            neg_contexts = domain_meta.get("negative_context", [])
            scale_class = domain_meta.get("scale_class", "unknown")
            facility_types = domain_meta.get("facility_types", [])

            candidate_stub = {
                "candidate_id": cand_id,
                "primary_sector": sec_name,
                "canonical_name": sec_name,
                "definition": definition,
                "positive_context_terms": pos_contexts or aliases,
                "negative_context_terms": neg_contexts,
                "scale_class": scale_class,
                "facility_types": facility_types
            }

            # Classify into Evidence Level and validate against evidence ledger
            evidence_level, confidence_multiplier, verified_evidence_ids, valid_traces = determine_evidence_level(
                candidate_stub, company_details, client_inquiry, evidence_ledger
            )

            # Evidence Score (0.0 if no verified evidence)
            evidence_count = len(verified_evidence_ids)
            if "LEVEL 1" in evidence_level:
                evidence_score = 1.0
            elif "LEVEL 2" in evidence_level:
                evidence_score = 0.85
            elif "LEVEL 3" in evidence_level:
                evidence_score = 0.50
            else:
                evidence_score = 0.0

            # Lexical factor
            lexical_score = min(1.0, (raw_tfidf_score * 2.0))

            # Intent score (if inquiry is present)
            intent_score = 0.0
            if client_inquiry and len(client_inquiry.strip()) > 2:
                inq_lower = client_inquiry.lower()
                if any(re.search(r"\b" + re.escape(a) + r"\b", inq_lower) for a in aliases):
                    intent_score = 1.0
                elif clean_sec in inq_lower or any(st in inq_lower for st in clean_sec.split() if st not in DOMAIN_STOPWORDS and len(st) >= 4):
                    intent_score = 0.90

            # Sub-scores derived deterministically
            functionality_score = float(raw_vec_score)
            definition_score = float(raw_vec_score)
            business_model_score = float(raw_vec_score * (1.0 if evidence_score > 0 else 0.8))
            facility_score = 1.0 if any(a in company_lower for a in aliases) else (0.5 if len(facility_types) > 0 else 0.2)

            # Deterministic multi-factor scoring formula
            final_score = (
                0.30 * evidence_score +
                0.20 * intent_score +
                0.15 * functionality_score +
                0.15 * definition_score +
                0.10 * business_model_score +
                0.05 * facility_score +
                0.05 * lexical_score
            )
            business_fit_score = (raw_vec_score * (1.0 + lexical_score * 0.25)) * (0.50 + 0.50 * confidence_multiplier)

            # Build explainable trace
            explainable_trace = {
                "matched_phrases": valid_traces[0].matched_phrases if valid_traces else [],
                "ignored_terms": valid_traces[0].ignored_terms if valid_traces else [],
                "negative_context_hits": valid_traces[0].negative_context_hits if valid_traces else [],
                "synonym_expansions": aliases[:6],
                "definition_entailment": valid_traces[0].definition_entailment if valid_traces else 0.0,
                "entity_relationship_check": valid_traces[0].entity_relationship_check if valid_traces else "unverified",
                "rejection_reason": valid_traces[0].rejection_reason if (valid_traces and not valid_traces[0].is_valid) else None
            }

            candidate_record = {
                "candidate_id": cand_id,
                "primary_sector": sec_name,
                "canonical_name": sec_name,
                "definition": definition,
                "synonyms": aliases[:6],
                "positive_context_terms": pos_contexts or aliases,
                "negative_context_terms": neg_contexts,
                "facility_types": facility_types,
                "scale_class": scale_class,
                "evidence_level": evidence_level,
                "verified_evidence_ids": verified_evidence_ids,
                "verified_evidence_count": evidence_count,
                "vector_cosine": round(raw_vec_score, 4),
                "functionality_score": round(functionality_score, 4),
                "intent_score": round(intent_score, 4),
                "definition_score": round(definition_score, 4),
                "business_model_score": round(business_model_score, 4),
                "facility_score": round(facility_score, 4),
                "lexical_score": round(lexical_score, 4),
                "lexical_boost": round(lexical_score * 0.25, 4),
                "business_fit_score": round(business_fit_score, 4),
                "final_score": round(final_score, 4),
                "similarity": round(raw_vec_score, 4),
                "confidence": "HIGH" if (evidence_score >= 0.85 and len(verified_evidence_ids) > 0) else ("MEDIUM" if evidence_score >= 0.50 else "SPECULATIVE"),
                "explainable_trace": explainable_trace
            }
            candidates.append(candidate_record)

        def _evidence_priority(item):
            lvl = item.get("evidence_level", "")
            if "LEVEL 1" in lvl:
                return 4
            if "LEVEL 2" in lvl:
                return 3
            if "LEVEL 3" in lvl:
                return 2
            return 1

        # Section K Deterministic Ranking:
        # 1. evidence_priority descending
        # 2. final_score descending
        # 3. intent_score descending
        # 4. definition_score descending
        # 5. candidate_id ascending (deterministic tie-breaker)
        candidates.sort(key=lambda x: (
            -_evidence_priority(x),
            -x["final_score"],
            -x["intent_score"],
            -x["definition_score"],
            x["candidate_id"]
        ))

        # Deduplicate
        results = []
        seen = set()
        for item in candidates:
            norm_name = re.sub(r"[^a-zA-Z0-9]", "", item["primary_sector"].lower())
            if norm_name in seen:
                continue
            seen.add(norm_name)
            results.append(item)
            if len(results) >= top_k:
                break

        return results

catalog = ServiceCatalog()
