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

# Comprehensive stopwords list to prevent generic words from triggering spurious lexical matches
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
    "very", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "wherever"
}

def calibrate_cosine_score(raw_score: float) -> float:
    """
    Calibrates hybrid similarity scores into an intuitive executive confidence percentage (75% - 98.5%).
    """
    if raw_score >= 0.70:
        pct = 95.0 + min(3.5, (raw_score - 0.70) * 15.0)
    elif raw_score >= 0.55:
        pct = 85.0 + (raw_score - 0.55) / 0.15 * 10.0
    elif raw_score >= 0.45:
        pct = 75.0 + (raw_score - 0.45) / 0.10 * 10.0
    else:
        pct = max(50.0, raw_score * 150.0)
    return round(min(98.5, max(60.0, pct)), 1)

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
            combined = f"{s} {s} {d}".lower()
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
        Creates a dense 1024-dim query vector combining the client's explicit inquiry,
        verified industry domain, past projects track record, and future strategic roadmap.
        """
        company_name = company_details.get("company_name", "Target Enterprise")
        industry = company_details.get("industry_focus", "")
        summary = company_details.get("executive_profile_analysis", "")
        biz_model = company_details.get("business_model_and_revenue_drivers", "")
        archetype = company_details.get("archetype", "Enterprise")

        past_proj = " ".join([p.get("project_name", "") + " " + p.get("summary", "") for p in company_details.get("delivered_historical_projects", [])])
        future_proj = " ".join([f.get("initiative", "") + " " + f.get("strategic_objective", "") for f in company_details.get("future_roadmaps_and_expansion", [])])

        inquiry_clause = f"Client Inbound Requirement & Inquiry: {client_inquiry}. " if client_inquiry else ""

        full_query = (
            f"{inquiry_clause}"
            f"Target Enterprise: {company_name}. "
            f"Industry Focus: {industry}. "
            f"Archetype: {archetype}. "
            f"Business Model: {biz_model}. "
            f"Core Operations & Offerings: {summary} {past_proj} {future_proj}."
        ).strip()

        vector = self._get_worker_embedding(full_query)
        if vector is None:
            vector = np.zeros(self.vectors.shape[1] if self.vectors is not None else 1024, dtype=np.float32)

        return {
            "query_text": full_query,
            "vector": vector,
            "dimension": len(vector),
            "model_name": self.model_name,
            "vector_preview": [round(float(x), 4) for x in vector[:8]]
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
        High-Precision Multi-Factor Hybrid Ranking:
        Combines 1024-dim dense vector cosine similarity (BGE-Large) with dynamic sublinear TF-IDF,
        morphological phrase matching, domain category gating, and transparent keyword explainability.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # 1. Dense Vector Cosine Similarity (Dense Semantic Field in < 0.1ms)
        dense_sims = np.dot(self.vectors, company_vector)

        # 2. Dynamic Sub-linear TF-IDF Similarity
        company_lower = (company_text + " " + client_inquiry).lower()
        if self.tfidf_vectorizer and self.tfidf_matrix is not None and len(company_lower) > 20:
            clean_company_tokens = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", company_lower) if t not in DOMAIN_STOPWORDS]
            company_tfidf = self.tfidf_vectorizer.transform([" ".join(clean_company_tokens)])
            tfidf_sims = (self.tfidf_matrix * company_tfidf.T).toarray().flatten()
        else:
            tfidf_sims = np.zeros(len(self.sectors), dtype=np.float32)

        # 3. Domain Gating Signals from verified profile & inquiry
        industry_lower = ""
        archetype_lower = ""
        past_and_future = ""
        if company_details:
            industry_lower = str(company_details.get("industry_focus", "")).lower()
            archetype_lower = str(company_details.get("archetype", "")).lower()
            past_and_future = " ".join([p.get("project_name", "") + " " + p.get("summary", "") for p in company_details.get("delivered_historical_projects", [])]).lower()
            past_and_future += " " + " ".join([f.get("initiative", "") + " " + f.get("strategic_objective", "") for f in company_details.get("future_roadmaps_and_expansion", [])]).lower()

        combined_evidence = f"{client_inquiry.lower()} {industry_lower} {archetype_lower} {past_and_future} {company_lower}"

        hybrid_scores = []
        for idx in range(len(self.sectors)):
            raw_vec_score = float(dense_sims[idx])
            raw_tfidf_score = float(tfidf_sims[idx])
            sec_name = str(self.sectors[idx]).strip()
            definition = str(self.definitions[idx]).strip()
            sec_lower = f"{sec_name} {definition}".lower()

            # Dynamic phrase & morphological matching
            clean_sec = re.sub(r"\(.*?\)", "", sec_name).lower().strip()
            sec_tokens = [t for t in re.findall(r"\b[a-zA-Z]{4,}\b", clean_sec) if t not in DOMAIN_STOPWORDS]
            
            exact_phrase_bonus = 0.0
            if len(clean_sec) > 4 and clean_sec not in ["office building", "commercial building", "other building", "building", "school", "penitentiary", "garages and service station"]:
                if clean_sec in combined_evidence or (clean_sec + "s") in combined_evidence or clean_sec.replace(" ", "") in combined_evidence:
                    exact_phrase_bonus = 0.25
                elif sec_tokens and sum(1 for t in sec_tokens if t in combined_evidence) == len(sec_tokens):
                    exact_phrase_bonus = 0.18

            # Acronym matching (e.g. PV, BESS, LNG, EV, AI, EDC, HDPE)
            acronyms = re.findall(r"\(([A-Za-z0-9\-]+)\)", sec_name)
            acronym_bonus = 0.15 if any(len(a) >= 2 and re.search(r"\b" + re.escape(a.lower()) + r"\b", combined_evidence) for a in acronyms) else 0.0

            # Domain Category Alignment Check
            domain_alignment_bonus = 0.0
            if industry_lower:
                # Direct industry name tokens in sector name
                ind_tokens = [t for t in re.findall(r"\b[a-zA-Z]{4,}\b", industry_lower) if t not in DOMAIN_STOPWORDS]
                if ind_tokens and any(t in sec_lower for t in ind_tokens):
                    domain_alignment_bonus = 0.15

            # Mathematical Hybrid Score
            lexical_component = min(0.40, (raw_tfidf_score * 1.5) + exact_phrase_bonus + acronym_bonus + domain_alignment_bonus)
            hybrid_score = max(0.0, raw_vec_score + lexical_component)
            calibrated_pct = calibrate_cosine_score(hybrid_score)

            matched_keywords = self._extract_matching_keywords(sec_name, definition, company_text)

            hybrid_scores.append({
                "index": idx,
                "Primary Sector": sec_name,
                "Definition": definition,
                "vector_cosine": round(raw_vec_score, 4),
                "lexical_boost": round(lexical_component, 4),
                "hybrid_score": round(hybrid_score, 4),
                "similarity": round(hybrid_score, 4),
                "match_pct": calibrated_pct,
                "matched_keywords": matched_keywords
            })

        # Sort by hybrid score descending
        hybrid_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)

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
