import os
import re
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_NPZ_PATH = BASE_DIR / "catalog_embeddings.npz"

# Comprehensive stopwords to prevent generic noise words from triggering spurious lexical matches
DOMAIN_STOPWORDS = {
    "plant", "facility", "facilities", "system", "systems", "production", "manufacturing", 
    "building", "buildings", "complex", "center", "centers", "infrastructure", "other", 
    "services", "service", "support", "management", "supporting", "integrated", "development",
    "long", "term", "site", "sites", "program", "programs", "business", "businesses", 
    "individual", "professional", "commercial", "industrial", "private", "growth", "scale",
    "home", "family", "built", "through", "with", "from", "that", "into", "these", "their",
    "about", "after", "again", "against", "because", "been", "before", "being", "below",
    "between", "both", "during", "each", "further", "having", "here", "more", "most",
    "once", "only", "same", "some", "such", "than", "then", "there", "they", "this", "those",
    "very", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "wherever",
    "data", "market", "markets", "care", "health", "specific", "track", "record", "project", "operations"
}

# Sectors that are strictly out-of-scope for standard commercial / middle-market enterprises
NON_COMMERCIAL_INSTITUTIONS = {
    "university", "school", "penitentiary", "animal shelter", "barrack",
    "armoury", "villa", "athletic track", "amusement facility", "prisons",
    "aircraft manufacturing plant", "nuclear power plant", "amusement park",
    "stadium", "sports complex", "crematorium", "cemetery"
}

def determine_evidence_level(
    sec_name: str, 
    definition: str, 
    company_details: Optional[dict], 
    client_inquiry: str = ""
) -> Tuple[str, float]:
    """
    Classifies a candidate offering into strict Ground-Truth Evidence Levels (1 to 5).
    Returns (level_label, confidence_multiplier).
    """
    clean_sec = re.sub(r"\(.*?\)", "", sec_name).lower().strip()
    sec_tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", clean_sec)) - DOMAIN_STOPWORDS

    industry_lower = str(company_details.get("industry_focus", "")).lower() if company_details else ""
    archetype_lower = str(company_details.get("archetype", "")).lower() if company_details else ""

    # 1. IMMEDIATE DISQUALIFICATION GATE (LEVEL 5)
    if clean_sec in NON_COMMERCIAL_INSTITUTIONS:
        is_inst = any(k in archetype_lower or k in industry_lower for k in ["university", "education", "school", "prison", "defense", "military", "aerospace", "sports"])
        if not is_inst:
            return "LEVEL 5 (Unsupported / Out-of-Scope)", 0.0

    # Heavy asset / mega-infrastructure mismatch check for middle-market funds
    if "private equity" in archetype_lower or "middle market" in industry_lower:
        if clean_sec in ["refinery", "oil refinery", "crude distillation unit", "blast furnace", "smelter", "nuclear power plant"]:
            return "LEVEL 5 (Unsupported / Scale Mismatch)", 0.0

    inquiry_lower = client_inquiry.lower()
    if inquiry_lower and len(inquiry_lower) > 3:
        if clean_sec in inquiry_lower or (sec_tokens and all(re.search(r"\b" + re.escape(t) + r"\b", inquiry_lower) for t in sec_tokens)):
            return "LEVEL 1 (Explicit Stated Requirement)", 1.0

    if not company_details:
        return "LEVEL 4 (Speculative / Semantic Only)", 0.40

    # 2. CHECK EXPLICIT CORE SECTOR (LEVEL 1)
    if clean_sec in industry_lower:
        return "LEVEL 1 (Explicit Core Sector)", 0.95
    if sec_tokens and len(sec_tokens) >= 2 and all(re.search(r"\b" + re.escape(t) + r"\b", industry_lower) for t in sec_tokens):
        return "LEVEL 1 (Explicit Core Sector)", 0.95

    # 3. CHECK VERIFIED PORTFOLIO CASE STUDIES AND OPERATIONS (LEVEL 2)
    past_text = " ".join([p.get("project_name", "") + " " + p.get("summary", "") for p in company_details.get("delivered_historical_projects", [])]).lower()
    active_text = " ".join([o.get("operation_name", "") + " " + o.get("details", "") for o in company_details.get("current_active_operations", [])]).lower()
    portfolio_text = f"{past_text} {active_text}"

    if clean_sec in portfolio_text:
        return "LEVEL 2 (Verified Portfolio Exposure)", 0.85
    if sec_tokens and len(sec_tokens) >= 2 and all(re.search(r"\b" + re.escape(t) + r"\b", portfolio_text) for t in sec_tokens):
        return "LEVEL 2 (Verified Portfolio Exposure)", 0.85

    # 4. CHECK STRATEGIC EXPANSION & ROADMAP (LEVEL 3)
    future_text = " ".join([f.get("initiative", "") + " " + f.get("strategic_objective", "") for f in company_details.get("future_roadmaps_and_expansion", [])]).lower()
    if clean_sec in future_text:
        return "LEVEL 3 (Strategic Roadmap Adjacency)", 0.70
    if sec_tokens and len(sec_tokens) >= 2 and all(re.search(r"\b" + re.escape(t) + r"\b", future_text) for t in sec_tokens):
        return "LEVEL 3 (Strategic Roadmap Adjacency)", 0.70

    return "LEVEL 4 (Speculative / Semantic Only)", 0.40

class ServiceCatalog:
    def __init__(self, npz_path=None):
        self.vectors = None        # shape (462, 1024)
        self.sectors = None        # array of 462 sector strings
        self.definitions = None    # array of 462 definition strings
        self.texts = None          # array of text strings
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
        self.sectors = data["sectors"]
        self.definitions = data["definitions"]
        self.texts = data["texts"]
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

    def _get_worker_embedding(self, text: str) -> np.ndarray:
        """Generates a 1024-dim dense vector using Cloudflare Workers AI with caching."""
        cache_key = text.strip().lower()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        if not self.worker_url:
            return None
        try:
            resp = requests.post(
                self.worker_url.rstrip("/") + "/ai/embed",
                json={"model": self.model_name, "text": [text]},
                headers={"Content-Type": "application/json"},
                timeout=25
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
        except Exception as e:
            print(f"[Embedding Error]: {e}")
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

        past_proj = " ".join([p.get("project_name", "") + " " + p.get("summary", "") for p in company_details.get("delivered_historical_projects", [])])
        future_proj = " ".join([f.get("initiative", "") + " " + f.get("strategic_objective", "") for f in company_details.get("future_roadmaps_and_expansion", [])])

        strategy_text = f"Enterprise: {company_name}. Archetype: {archetype}. Core Focus Sectors & Strategy: {industry}. Business Model: {biz_model}."
        portfolio_text = f"Portfolio Operations & Track Record: {summary} {past_proj} {future_proj}."

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
        company_lower = company_text.lower()
        combined_sector = f"{sector_name} {definition}".lower()
        
        tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", combined_sector))
        informative_tokens = tokens - DOMAIN_STOPWORDS
        
        matches = []
        for t in informative_tokens:
            if re.search(r"\b" + re.escape(t) + r"\b", company_lower):
                matches.append(t)
        return sorted(matches[:6])

    def match_company_vector(
        self,
        company_vector: np.ndarray,
        company_text: str = "",
        company_details: Optional[dict] = None,
        client_inquiry: str = "",
        top_k: int = 15
    ) -> list:
        """
        Evidence-Grounded Multi-Factor Matching:
        Combines 1024-dim dense vector cosine similarity with Multiplicative Lexical Gating,
        Ground-Truth Evidence Level classification (Levels 1 to 5), and transparent rejection of unsupported sectors.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # 1. Dense Vector Cosine Similarity (Dense Semantic Field)
        dense_sims = np.dot(self.vectors, company_vector)

        # 2. Dynamic Sub-linear TF-IDF Similarity
        company_lower = (company_text + " " + client_inquiry).lower()
        if self.tfidf_vectorizer and self.tfidf_matrix is not None and len(company_lower) > 20:
            clean_company_tokens = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", company_lower) if t not in DOMAIN_STOPWORDS]
            company_tfidf = self.tfidf_vectorizer.transform([" ".join(clean_company_tokens)])
            tfidf_sims = (self.tfidf_matrix * company_tfidf.T).toarray().flatten()
        else:
            tfidf_sims = np.zeros(len(self.sectors), dtype=np.float32)

        hybrid_scores = []
        for idx in range(len(self.sectors)):
            raw_vec_score = float(dense_sims[idx])
            raw_tfidf_score = float(tfidf_sims[idx])
            sec_name = str(self.sectors[idx]).strip()
            definition = str(self.definitions[idx]).strip()
            clean_sec = re.sub(r"\(.*?\)", "", sec_name).lower().strip()

            # Classify into Evidence Level (1 to 5)
            evidence_level, confidence_multiplier = determine_evidence_level(sec_name, definition, company_details, client_inquiry)

            # Skip Level 5 (Unsupported / Out-of-Scope) sectors completely
            if confidence_multiplier == 0.0:
                continue

            # Multiplicative Lexical Factor
            lexical_factor = min(0.25, (raw_tfidf_score * 1.2))
            if clean_sec in company_lower or (clean_sec + "s") in company_lower:
                lexical_factor = min(0.25, lexical_factor + 0.10)

            # Evidence-Grounded Multi-Factor Business Fit Score
            # Multiplicative scaling ensures lexical tokens amplify, but cannot manufacture, vector relevance
            base_score = raw_vec_score * (1.0 + lexical_factor)
            business_fit_score = base_score * (0.50 + 0.50 * confidence_multiplier)

            matched_keywords = self._extract_matching_keywords(sec_name, definition, company_text)

            hybrid_scores.append({
                "index": idx,
                "Primary Sector": sec_name,
                "Definition": definition,
                "vector_cosine": round(raw_vec_score, 4),
                "lexical_boost": round(lexical_factor, 4),
                "business_fit_score": round(business_fit_score, 4),
                "similarity": round(raw_vec_score, 4),
                "evidence_level": evidence_level,
                "confidence": "HIGH" if confidence_multiplier >= 0.85 else ("MEDIUM" if confidence_multiplier >= 0.70 else "SPECULATIVE"),
                "matched_keywords": matched_keywords
            })

        # Sort by business fit score descending
        hybrid_scores.sort(key=lambda x: x["business_fit_score"], reverse=True)

        # Deduplicate and return top_k
        results = []
        seen = set()
        for item in hybrid_scores:
            norm_name = re.sub(r"[^a-zA-Z0-9]", "", item["Primary Sector"].lower())
            if norm_name in seen:
                continue
            seen.add(norm_name)

            results.append(item)
            if len(results) >= top_k:
                break

        return results

catalog = ServiceCatalog()
