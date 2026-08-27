import os
import re
import requests
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_NPZ_PATH = BASE_DIR / "catalog_embeddings.npz"

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
        """Generates a 1024-dim dense vector for the target company using Cloudflare Workers AI."""
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
        """Extracts and creates dense vector embedding for any target company website."""
        company_name = company_details.get("company_name", "Target Company")
        industry = company_details.get("industry_focus", "Industrial Enterprise")
        summary = company_details.get("executive_profile_analysis", "") or company_details.get("executive_summary", "")
        needs = company_details.get("expectations_and_needs_narrative", "") or " ".join(company_details.get("expectations_and_needs", []))
        friction = company_details.get("operational_friction_analysis", "") or " ".join(company_details.get("core_friction_points", []))

        # Comprehensive semantic query representation
        query_text = (
            f"Company: {company_name}. "
            f"Industry Focus: {industry}. "
            f"Executive Overview & Operations: {summary}. "
            f"Target Infrastructure Scope & Strategic Requirements: {needs}. {friction}"
        ).strip()

        vector = self._get_worker_embedding(query_text)
        
        # Fallback if offline/network timeout
        if vector is None:
            vector = np.zeros(self.vectors.shape[1] if self.vectors is not None else 1024, dtype=np.float32)

        return {
            "query_text": query_text,
            "vector": vector,
            "dimension": len(vector),
            "model_name": self.model_name,
            "vector_preview": [round(float(x), 4) for x in vector[:8]]
        }

    def match_company_vector(self, company_vector: np.ndarray, top_k: int = 3) -> list:
        """Performs vector cosine similarity search across all 462 pre-computed catalog embeddings."""
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # Matrix Dot Product on normalized vectors = exact Cosine Similarity in < 0.1ms
        sims = np.dot(self.vectors, company_vector)
        sorted_indices = np.argsort(sims)[::-1]

        results = []
        seen_sectors = set()

        for idx in sorted_indices:
            score = float(sims[idx])
            sector_name = str(self.sectors[idx]).strip()
            definition = str(self.definitions[idx]).strip()

            norm_name = re.sub(r"[^a-zA-Z0-9]", "", sector_name.lower())
            if norm_name in seen_sectors:
                continue

            seen_sectors.add(norm_name)
            
            # Formatted percentage score
            match_pct = round(max(0.0, min(score * 100, 99.0)), 1)

            results.append({
                "Primary Sector": sector_name,
                "Definition": definition,
                "similarity": round(score, 4),
                "match_pct": match_pct
            })

            if len(results) >= top_k:
                break

        return results

catalog = ServiceCatalog()
