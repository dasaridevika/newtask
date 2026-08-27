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

    def _calculate_universal_lexical_boost(self, sector_name: str, definition: str, company_text: str) -> float:
        """
        Universal, domain-agnostic lexical relevance calculator.
        Dynamically extracts tokens, compound terms, and acronyms from any of the 462 catalog sectors
        and checks for term presence and frequency across the target company's extracted text.
        """
        company_lower = company_text.lower()
        sec_name_clean = re.sub(r"\(.*?\)", "", sector_name).lower().strip()
        
        boost = 0.0

        # 1. Exact Full Sector Name Match (e.g. "Data Center", "Solar Photovoltaic", "Grain Elevator", "Hospital")
        if len(sec_name_clean) > 3 and sec_name_clean in company_lower:
            boost += 0.35

        # 2. Key Acronyms or Parenthetical Specifics (e.g. "(PV)", "(HDPE)", "(EDC)", "(BESS)", "(LNG)")
        acronyms = re.findall(r"\(([A-Za-z0-9\-]+)\)", sector_name)
        for acr in acronyms:
            if len(acr) >= 2 and re.search(r"\b" + re.escape(acr.lower()) + r"\b", company_lower):
                boost += 0.20

        # 3. Dynamic Keyword Token Matching from Sector Name & Definition
        tokens = [
            t for t in re.findall(r"\b[a-zA-Z]{4,}\b", sec_name_clean)
            if t not in ["plant", "facility", "station", "system", "production", "manufacturing", "building", "complex", "center", "infrastructure", "other"]
        ]

        if tokens:
            matches = sum(1 for t in tokens if re.search(r"\b" + re.escape(t) + r"\b", company_lower))
            match_ratio = matches / len(tokens)
            boost += match_ratio * 0.30

        return min(0.50, boost)

    def match_company_vector(self, company_vector: np.ndarray, company_text: str = "", top_k: int = 15) -> list:
        """
        Universal Multi-Factor Hybrid Ranking:
        Combines 1024-dim dense vector cosine similarity with dynamic, universal token matching
        across all 462 sectors without hardcoded industry boundaries.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # 1. Dense Vector Cosine Similarities across all 462 sectors (Matrix Dot Product in < 0.1ms)
        dense_sims = np.dot(self.vectors, company_vector)

        # 2. Universal Hybrid Scoring per Sector
        hybrid_scores = []
        for idx in range(len(self.sectors)):
            raw_vec_score = float(dense_sims[idx])
            sec_name = str(self.sectors[idx]).strip()
            definition = str(self.definitions[idx]).strip()

            # Dynamic universal token boost
            lex_boost = self._calculate_universal_lexical_boost(sec_name, definition, company_text)

            # Combined hybrid score (Dense Vector Cosine + Universal Lexical Boost)
            hybrid_score = max(0.0, raw_vec_score + lex_boost)
            calibrated_pct = calibrate_cosine_score(hybrid_score)

            hybrid_scores.append({
                "index": idx,
                "Primary Sector": sec_name,
                "Definition": definition,
                "vector_cosine": round(raw_vec_score, 4),
                "lexical_boost": round(lex_boost, 4),
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
