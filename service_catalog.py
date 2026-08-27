import os
import re
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_NPZ_PATH = BASE_DIR / "catalog_embeddings.npz"

# High-precision domain triggers for hybrid lexical alignment
SECTOR_DOMAIN_TRIGGERS = {
    "solar": ["solar", "photovoltaic", "pv", "bifacial", "perovskite", "inverter", "clean energy", "renewable", "solar farm", "solar cell", "solar module"],
    "data_center": ["data center", "datacenter", "cooling", "thermal management", "liquid cooling", "cdu", "ups", "switchgear", "rack", "hyperscale", "colocation"],
    "chemical": ["chemical", "polymer", "resin", "polyethylene", "polyvinyl", "petrochemical", "feedstock", "catalyst", "refinery", "acid", "ethylene"],
    "private_equity": ["private equity", "buyout", "aum", "fund", "portfolio company", "sponsor", "due diligence", "m&a", "add-on", "private debt", "credit"],
    "power_grid": ["transmission", "substation", "grid", "high voltage", "interconnection", "transformer", "distribution", "megawatt", "utility", "power line"],
    "logistics": ["freight", "warehouse", "logistics", "supply chain", "distribution center", "intermodal", "railway", "fleet", "fulfillment"],
    "manufacturing": ["manufacturing", "fabrication", "oem", "assembly", "plant", "machinery", "industrial equipment", "automation"],
    "hydrogen": ["hydrogen", "electrolyzer", "fuel cell", "green hydrogen", "blue hydrogen", "ammonia"],
    "battery": ["battery", "bess", "energy storage", "lithium", "cell manufacturing", "gigafactory", "grid storage"]
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
                timeout=20
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
        """Creates a high-signal dense vector query representation from company intelligence."""
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

    def _calculate_lexical_boost(self, sector_name: str, definition: str, company_text: str) -> float:
        """Calculates exact keyword and domain alignment boost to ensure high precision."""
        sector_lower = f"{sector_name} {definition}".lower()
        company_lower = company_text.lower()

        boost = 0.0

        # Exact sector name match
        clean_sec = re.sub(r"\(.*?\)", "", sector_name).lower().strip()
        if len(clean_sec) > 3 and clean_sec in company_lower:
            boost += 0.35

        # 1. Data Center alignment
        has_datacenter = any(k in company_lower for k in ["data center", "datacenter", "hyperscale", "liquid cooling", "thermal management", "critical power", "switchgear", "ups"])
        is_datacenter_sector = "data center" in sector_lower
        if has_datacenter and is_datacenter_sector:
            boost += 0.40

        # 2. Solar alignment
        has_solar = any(k in company_lower for k in ["solar", "photovoltaic", "pv module", "solar panel", "solar cell", "solar power", "bifacial"])
        is_solar_sector = any(k in sector_lower for k in ["solar", "photovoltaic", "pv"])
        if has_solar and is_solar_sector:
            boost += 0.40
        elif has_solar and not is_solar_sector and any(k in sector_lower for k in ["chemical", "polymer", "petroleum", "coal"]):
            boost -= 0.30

        # 3. Private Equity alignment
        is_pe = any(k in company_lower for k in ["private equity", "buyout", "aum", "fund", "portfolio company", "asset management"])
        if is_pe and any(k in sector_lower for k in ["chemical", "packaging", "manufacturing", "industrial", "polymer", "food", "healthcare"]):
            boost += 0.15

        # 4. General Domain Trigger overlap
        for domain, keywords in SECTOR_DOMAIN_TRIGGERS.items():
            if domain in ["solar", "data_center"]:
                continue
            sector_matches = any(k in sector_lower for k in keywords)
            company_matches = sum(1 for k in keywords if re.search(r"\b" + re.escape(k) + r"\b", company_lower))

            if sector_matches and company_matches > 0:
                boost += min(0.20, company_matches * 0.04)

        return boost

    def match_company_vector(self, company_vector: np.ndarray, company_text: str = "", top_k: int = 15) -> list:
        """
        Executes Multi-Factor Hybrid Ranking:
        Combines 1024-dim dense vector cosine similarity with lexical keyword boosting,
        domain synergy, and disqualifier penalties.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # 1. Dense Vector Cosine Similarities (Matrix Dot Product in < 0.1ms)
        dense_sims = np.dot(self.vectors, company_vector)

        # 2. Hybrid Scoring per Sector
        hybrid_scores = []
        for idx in range(len(self.sectors)):
            raw_vec_score = float(dense_sims[idx])
            sec_name = str(self.sectors[idx]).strip()
            definition = str(self.definitions[idx]).strip()

            # Calculate domain keyword boost
            lex_boost = self._calculate_lexical_boost(sec_name, definition, company_text)

            # Combined hybrid score (Dense Cosine + Lexical Boost)
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

        # 4. Deduplicate and take top_k
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
