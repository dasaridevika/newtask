import os
import re
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_NPZ_PATH = BASE_DIR / "catalog_embeddings.npz"

# Comprehensive stopwords to prevent generic noise words from triggering spurious lexical matches
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
    "launch", "stories", "success", "driven", "training", "academic", "university", "universities", "school"
}

# Ambiguous single terms requiring co-occurring physical domain anchor terms
AMBIGUOUS_SECTORS = {
    "overhead": {"power", "transmission", "grid", "cable", "cables", "pole", "poles", "aerial", "structure", "high-voltage", "line", "lines"},
    "road": {"highway", "pavement", "asphalt", "civil", "transportation", "toll", "traffic", "construction"},
    "university": {"higher education", "undergraduate", "postgraduate", "campus", "academic institution", "faculty", "degree"},
    "aircraft": {"hangar", "aerospace", "aviation", "fuselage", "boeing", "airbus", "plane", "aircraft"},
    "office": {"commercial real estate", "headquarters", "tenant", "cre", "workspaces", "office building"}
}

def determine_evidence_level(
    sec_name: str, 
    definition: str, 
    company_details: Optional[dict], 
    client_inquiry: str = "",
    evidence_ledger: Optional[List[Any]] = None
) -> Tuple[str, float, List[str]]:
    """
    Classifies a candidate offering into Ground-Truth Evidence Levels dynamically
    with strict multi-token phrase validation and polysemy disambiguation.
    Returns (level_label, confidence_multiplier, verified_evidence_ids).
    """
    clean_sec = re.sub(r"\(.*?\)", "", sec_name).lower().strip()
    clean_sec_norm = clean_sec.replace(" ", "").replace("-", "")
    sec_tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", clean_sec)) - DOMAIN_STOPWORDS

    verified_evidence_ids = []

    # Strict multi-token and phrase validation against evidence quotes
    if evidence_ledger:
        for ev in evidence_ledger:
            ev_dict = ev if isinstance(ev, dict) else (ev.to_dict() if hasattr(ev, "to_dict") else {})
            ev_id = ev_dict.get("evidence_id", "")
            norm_quote = ev_dict.get("normalized_text", "")
            
            # Guard against ambiguous words
            if clean_sec in AMBIGUOUS_SECTORS:
                required_anchors = AMBIGUOUS_SECTORS[clean_sec]
                if clean_sec in norm_quote and any(re.search(r"\b" + re.escape(a) + r"\b", norm_quote) for a in required_anchors):
                    if ev_id and ev_id not in verified_evidence_ids:
                        verified_evidence_ids.append(ev_id)
                continue

            # Exact multi-word phrase match (e.g. "warehouse distribution", "solar photovoltaic", "health care building")
            if len(clean_sec.split()) >= 2 and clean_sec in norm_quote:
                if ev_id and ev_id not in verified_evidence_ids:
                    verified_evidence_ids.append(ev_id)
            # Distinctive multi-token match: requires AT LEAST 2 non-stopword tokens or a highly unique domain anchor
            elif len(sec_tokens) >= 2:
                matched_toks = [st for st in sec_tokens if re.search(r"\b" + re.escape(st) + r"\b", norm_quote)]
                if len(matched_toks) >= 2:
                    if ev_id and ev_id not in verified_evidence_ids:
                        verified_evidence_ids.append(ev_id)
            elif len(sec_tokens) == 1:
                single_tok = list(sec_tokens)[0]
                if len(single_tok) >= 6 and re.search(r"\b" + re.escape(single_tok) + r"\b", norm_quote):
                    # Check definition context
                    def_tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", definition.lower())) - DOMAIN_STOPWORDS
                    if any(re.search(r"\b" + re.escape(dt) + r"\b", norm_quote) for dt in def_tokens if len(dt) >= 5):
                        if ev_id and ev_id not in verified_evidence_ids:
                            verified_evidence_ids.append(ev_id)

    # 1. CHECK INBOUND CLIENT INQUIRY (LEVEL 1)
    if client_inquiry:
        inq_lower = client_inquiry.lower()
        inq_norm = re.sub(r"[^a-zA-Z0-9]", "", inq_lower)
        if clean_sec in inq_lower and len(clean_sec) >= 4:
            return "LEVEL 1 (Explicit Stated Requirement)", 0.95, verified_evidence_ids or ["inquiry_direct_stated"]
        if clean_sec_norm in inq_norm and len(clean_sec_norm) >= 5:
            return "LEVEL 1 (Explicit Stated Requirement)", 0.95, verified_evidence_ids or ["inquiry_direct_stated"]
        if sec_tokens and any(re.search(r"\b" + re.escape(st) + r"\b", inq_lower) for st in sec_tokens if len(st) >= 4):
            return "LEVEL 1 (Explicit Stated Requirement)", 0.95, verified_evidence_ids or ["inquiry_direct_stated"]

    if not company_details:
        return "LEVEL 4 (Speculative / Semantic Only)", 0.40, []

    # 2. CHECK EXPLICIT PORTFOLIO TARGET SECTORS (LEVEL 1 / 2)
    target_secs = company_details.get("portfolio_target_sectors", [])
    for ts in target_secs:
        ts_clean = re.sub(r"[^a-zA-Z0-9\s]", "", ts).lower().strip()
        ts_norm = ts_clean.replace(" ", "")
        
        # Exact match of sector phrase
        if clean_sec == ts_clean or clean_sec_norm == ts_norm:
            return "LEVEL 1 (Explicit Stated Focus)", 0.95, verified_evidence_ids or ["profile_stated_focus"]

        # Check compound/normalized overlap (e.g. healthcare vs health care, telecom vs telecommunication)
        if sec_tokens:
            for st in sec_tokens:
                if len(st) >= 4 and (re.search(r"\b" + re.escape(st) + r"\b", ts_clean) or (len(st) >= 5 and st in ts_norm)):
                    return "LEVEL 2 (Verified Portfolio Exposure)", 0.90, verified_evidence_ids or ["profile_target_sector"]
            
        ts_tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", ts_clean)) - DOMAIN_STOPWORDS
        if ts_tokens:
            for tt in ts_tokens:
                if len(tt) >= 4 and (re.search(r"\b" + re.escape(tt) + r"\b", clean_sec) or (len(tt) >= 5 and tt in clean_sec_norm)):
                    return "LEVEL 2 (Verified Portfolio Exposure)", 0.90, verified_evidence_ids or ["profile_target_sector"]

    # 3. CHECK EXPLICIT CORE INDUSTRY FOCUS (LEVEL 1)
    industry_lower = str(company_details.get("industry_focus", "")).lower()
    if clean_sec == industry_lower:
        return "LEVEL 1 (Explicit Core Sector)", 0.95, verified_evidence_ids or ["industry_core_focus"]
    if sec_tokens and any(re.search(r"\b" + re.escape(st) + r"\b", industry_lower) for st in sec_tokens if len(st) >= 5):
        return "LEVEL 1 (Explicit Core Sector)", 0.95, verified_evidence_ids or ["industry_core_focus"]

    # 4. CHECK VERIFIED PORTFOLIO CASE STUDIES AND OPERATIONS (LEVEL 2)
    past_text = " ".join([p.get("project_name", "") + " " + p.get("summary", "") for p in company_details.get("delivered_historical_projects", [])]).lower()
    active_text = " ".join([o.get("operation_name", "") + " " + o.get("details", "") for o in company_details.get("current_active_operations", [])]).lower()
    portfolio_text = f"{past_text} {active_text}"

    if clean_sec in portfolio_text and len(clean_sec) >= 5:
        return "LEVEL 2 (Verified Portfolio Exposure)", 0.88, verified_evidence_ids or ["portfolio_case_study"]
    if sec_tokens and len(sec_tokens) >= 2 and all(re.search(r"\b" + re.escape(st) + r"\b", portfolio_text) for st in sec_tokens):
        return "LEVEL 2 (Verified Portfolio Exposure)", 0.88, verified_evidence_ids or ["portfolio_case_study"]

    # 5. CHECK STRATEGIC EXPANSION & ROADMAP (LEVEL 3)
    future_text = " ".join([f.get("initiative", "") + " " + f.get("strategic_objective", "") for f in company_details.get("future_roadmaps_and_expansion", [])]).lower()
    if clean_sec in future_text and len(clean_sec) >= 5:
        return "LEVEL 3 (Strategic Roadmap Adjacency)", 0.70, verified_evidence_ids or ["future_roadmap"]
    if sec_tokens and len(sec_tokens) >= 2 and all(re.search(r"\b" + re.escape(t) + r"\b", future_text) for t in sec_tokens):
        return "LEVEL 3 (Strategic Roadmap Adjacency)", 0.70, verified_evidence_ids or ["future_roadmap"]

    # If evidence ledger found high-confidence multi-token matches
    if verified_evidence_ids:
        return "LEVEL 2 (Verified Portfolio Exposure)", 0.85, verified_evidence_ids

    return "LEVEL 4 (Speculative / Semantic Only)", 0.40, []

    return "LEVEL 4 (Speculative / Semantic Only)", 0.40, []

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
            combined = f"{s} {d}".lower()
            tokens = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", combined) if t not in DOMAIN_STOPWORDS]
            cleaned_corpus.append(" ".join(tokens))

        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=6000
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
            except Exception as e:
                if attempt == 2:
                    print(f"[Embedding Error]: {e}")
                time.sleep(1.0 * (attempt + 1))
        return None

    def embed_company(self, company_details: dict, scraped_text: str = "", client_inquiry: str = "") -> dict:
        """
        Multi-Vector Representation Architecture:
        Generates distinct semantic vectors for Investment Strategy, Portfolio Operations,
        and Inbound Inquiries, then creates an L2-normalized weighted composite vector.
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

        if client_inquiry and len(client_inquiry.strip()) > 3:
            inq_vec = self._get_worker_embedding(f"Specific Client Inquiry & Stated Requirement: {client_inquiry}")
            if inq_vec is None:
                inq_vec = np.zeros(dim, dtype=np.float32)
            composite = 0.45 * strat_vec + 0.35 * port_vec + 0.20 * inq_vec
        else:
            composite = 0.55 * strat_vec + 0.45 * port_vec

        norm = np.linalg.norm(composite)
        normalized_vector = composite / (norm if norm > 0 else 1e-10)

        return {
            "query_text": f"{strategy_text} {portfolio_text}",
            "vector": normalized_vector,
            "dimension": len(normalized_vector),
            "model_name": self.model_name
        }

    def _extract_matching_keywords(self, sector_name: str, definition: str, company_text: str) -> List[str]:
        """Identifies specific domain keywords found in both the catalog sector and the evidence."""
        if not company_text:
            return []
        company_lower = company_text.lower()
        combined_sector = f"{sector_name} {definition}".lower()
        tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", combined_sector)) - DOMAIN_STOPWORDS
        matches = [t for t in tokens if re.search(r"\b" + re.escape(t) + r"\b", company_lower)]
        return sorted(matches[:6])

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
        Evidence-First Deterministic Matching:
        Combines 1024-dim dense vector cosine similarity with multi-factor scoring
        and cross-links directly with verifiable evidence IDs.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # 1. Dense Vector Cosine Similarity
        dense_sims = np.dot(self.vectors, company_vector)

        # 2. Dynamic TF-IDF Lexical Similarity
        company_lower = (company_text + " " + client_inquiry).lower()
        if self.tfidf_vectorizer and self.tfidf_matrix is not None and len(company_lower) > 20:
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

            # Classify into Evidence Level and extract verified evidence IDs
            evidence_level, confidence_multiplier, verified_evidence_ids = determine_evidence_level(
                sec_name, definition, company_details, client_inquiry, evidence_ledger
            )

            # Lexical factor
            lexical_factor = min(0.25, (raw_tfidf_score * 1.2))
            if clean_sec in company_lower or (clean_sec + "s") in company_lower:
                lexical_factor = min(0.25, lexical_factor + 0.10)

            # Intent score (if inquiry is present)
            intent_score = 0.0
            if client_inquiry:
                inq_lower = client_inquiry.lower()
                if clean_sec in inq_lower or any(st in inq_lower for st in clean_sec.split()):
                    intent_score = 0.95

            # Multi-factor business fit score
            base_score = raw_vec_score * (1.0 + lexical_factor)
            business_fit_score = base_score * (0.50 + 0.50 * confidence_multiplier)
            final_score = 0.60 * business_fit_score + 0.20 * raw_vec_score + 0.20 * (1.0 if len(verified_evidence_ids) > 0 else 0.40)

            matched_keywords = self._extract_matching_keywords(sec_name, definition, company_text)

            candidate_record = {
                "candidate_id": cand_id,
                "primary_sector": sec_name,
                "canonical_name": sec_name,
                "definition": definition,
                "evidence_level": evidence_level,
                "verified_evidence_ids": verified_evidence_ids,
                "verified_evidence_count": len(verified_evidence_ids),
                "vector_cosine": round(raw_vec_score, 4),
                "lexical_score": round(lexical_factor, 4),
                "lexical_boost": round(lexical_factor, 4),
                "intent_score": round(intent_score, 4),
                "definition_score": round(raw_vec_score, 4),
                "business_fit_score": round(business_fit_score, 4),
                "final_score": round(final_score, 4),
                "similarity": round(raw_vec_score, 4),
                "confidence": "HIGH" if confidence_multiplier >= 0.85 else ("MEDIUM" if confidence_multiplier >= 0.70 else "SPECULATIVE"),
                "matched_keywords": matched_keywords
            }
            candidates.append(candidate_record)

        def _evidence_tier(item):
            lvl = item.get("evidence_level", "")
            if "LEVEL 1" in lvl:
                return 1
            if "LEVEL 2" in lvl:
                return 2
            if "LEVEL 3" in lvl:
                return 3
            return 4

        # Sort primarily by evidence tier (Level 1 -> 2 -> 3 -> 4) and secondarily by final_score descending
        candidates.sort(key=lambda x: (_evidence_tier(x), -x["final_score"]))

        # Deduplicate by canonical name
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
