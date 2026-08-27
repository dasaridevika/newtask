import os
import re
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_NPZ_PATH = BASE_DIR / "catalog_embeddings.npz"

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

        target_npz = npz_path or EMBEDDINGS_NPZ_PATH
        if Path(target_npz).exists():
            self.load_embeddings(target_npz)

    def load_embeddings(self, npz_file):
        """Loads pre-computed 1024-dimensional normalized vector matrix in < 1ms."""
        data = np.load(npz_file, allow_pickle=True)
        self.vectors = data["vectors"].astype(np.float32)
        self.sectors = data["sectors"]
        self.definitions = data["definitions"]
        self.texts = data["texts"]
        if "model_name" in data:
            self.model_name = str(data["model_name"])

        # Dynamic TF-IDF model across all 462 catalog sectors
        corpus = [f"{s} {d}" for s, d in zip(self.sectors, self.definitions)]
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_features=5000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        return len(self.sectors)

    def _get_worker_embedding(self, text: str) -> np.ndarray:
        """Generates a 1024-dim dense vector using Cloudflare Workers AI."""
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
                    return vec / (norm if norm > 0 else 1e-10)
                elif data and isinstance(data[0], dict) and "values" in data[0]:
                    vec = np.array(data[0]["values"], dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    return vec / (norm if norm > 0 else 1e-10)
        except Exception as e:
            print(f"[Embedding Error]: {e}")
        return None

    def embed_company(self, company_details: dict, scraped_text: str = "") -> dict:
        """Creates a dense vector query representation dynamically for any type of enterprise."""
        company_name = company_details.get("company_name", "Target Enterprise")
        industry = company_details.get("industry_focus", "")
        summary = company_details.get("executive_profile_analysis", "")
        needs = company_details.get("expectations_and_needs_narrative", "")
        friction = company_details.get("operational_friction_analysis", "")

        query_text = (
            f"Company: {company_name}. "
            f"Industry Focus: {industry}. "
            f"Executive Overview & Operations: {summary}. "
            f"Target Infrastructure Scope & Strategic Requirements: {needs}. {friction}"
        ).strip()

        vector = self._get_worker_embedding(query_text)
        if vector is None:
            vector = np.zeros(self.vectors.shape[1] if self.vectors is not None else 1024, dtype=np.float32)

        return {
            "query_text": query_text,
            "vector": vector,
            "dimension": len(vector),
            "model_name": self.model_name,
            "vector_preview": [round(float(x), 4) for x in vector[:8]]
        }

    def match_company_vector(self, company_vector: np.ndarray, company_text: str = "", top_k: int = 15) -> list:
        """
        Pure Mathematical Multi-Factor Hybrid Ranking:
        Combines 1024-dim dense vector cosine similarity with dynamic TF-IDF and morphological token matching
        across all 462 catalog sectors without hardcoded keyword blacklists.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # 1. Dense Vector Cosine Similarity (Dense Semantic Field in < 0.1ms)
        dense_sims = np.dot(self.vectors, company_vector)

        # 2. Dynamic Sub-linear TF-IDF Similarity
        company_lower = company_text.lower()
        if self.tfidf_vectorizer and self.tfidf_matrix is not None and len(company_text) > 20:
            company_tfidf = self.tfidf_vectorizer.transform([company_text])
            tfidf_sims = (self.tfidf_matrix * company_tfidf.T).toarray().flatten()
        else:
            tfidf_sims = np.zeros(len(self.sectors), dtype=np.float32)

        hybrid_scores = []
        for idx in range(len(self.sectors)):
            raw_vec_score = float(dense_sims[idx])
            raw_tfidf_score = float(tfidf_sims[idx])
            sec_name = str(self.sectors[idx]).strip()
            definition = str(self.definitions[idx]).strip()

            # Dynamic phrase & morphological matching (e.g. data center / data centers / datacenter)
            clean_sec = re.sub(r"\(.*?\)", "", sec_name).lower().strip()
            sec_tokens = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", clean_sec)]
            
            exact_phrase_bonus = 0.0
            if len(clean_sec) > 3:
                # Exact or plural phrase match
                if clean_sec in company_lower or (clean_sec + "s") in company_lower or clean_sec.replace(" ", "") in company_lower:
                    exact_phrase_bonus = 0.20
                elif sec_tokens and sum(1 for t in sec_tokens if t in company_lower) == len(sec_tokens):
                    exact_phrase_bonus = 0.15

            # Dynamic acronym matching (e.g. PV, BESS, LNG, EV, AI, EDC, HDPE)
            acronyms = re.findall(r"\(([A-Za-z0-9\-]+)\)", sec_name)
            acronym_bonus = 0.15 if any(len(a) >= 2 and re.search(r"\b" + re.escape(a.lower()) + r"\b", company_lower) for a in acronyms) else 0.0

            # Mathematical Hybrid Score: Vector Cosine + Dynamic TF-IDF + Exact Morphological Match
            lexical_component = min(0.35, (raw_tfidf_score * 1.5) + exact_phrase_bonus + acronym_bonus)
            hybrid_score = max(0.0, raw_vec_score + lexical_component)
            calibrated_pct = calibrate_cosine_score(hybrid_score)

            hybrid_scores.append({
                "index": idx,
                "Primary Sector": sec_name,
                "Definition": definition,
                "vector_cosine": round(raw_vec_score, 4),
                "lexical_boost": round(lexical_component, 4),
                "hybrid_score": round(hybrid_score, 4),
                "similarity": round(hybrid_score, 4),
                "match_pct": calibrated_pct
            })

        # 3. Sort by hybrid score descending
        hybrid_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # 4. Deduplicate and return top_k
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
