import os
import re
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent

class ServiceCatalog:
    def __init__(self, file_path=None):
        self.df = None
        self.embeddings = None
        self.embedding_model_name = os.getenv("CF_EMBEDDING_MODEL", "@cf/baai/bge-large-en-v1.5")
        self.worker_url = os.getenv("CLOUDFLARE_WORKER_URL", "https://lead-research-ai-worker.devika-worker.workers.dev")
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        
        target_path = file_path
        if not target_path:
            csv_path = BASE_DIR / "primary_sector_with_definitions.csv"
            if csv_path.exists():
                target_path = csv_path
            else:
                user_files = [f for f in BASE_DIR.glob("*.*") if f.suffix.lower() in [".csv", ".xlsx"] and not f.name.startswith("~$")]
                if user_files:
                    target_path = user_files[0]
                
        if target_path and Path(target_path).exists():
            self.load(target_path)

    def _get_single_worker_embedding(self, text: str, model_name: str) -> np.ndarray:
        """Embeds single company profile via Cloudflare Workers AI with fast timeout."""
        if not self.worker_url:
            return None
        try:
            resp = requests.post(
                self.worker_url.rstrip("/") + "/ai/embed",
                json={"model": model_name, "text": [text]},
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data and isinstance(data[0], list):
                    return np.array(data[0], dtype=np.float32)
                elif data and isinstance(data[0], dict) and "values" in data[0]:
                    return np.array(data[0]["values"], dtype=np.float32)
        except Exception:
            pass
        return None

    def load(self, file_source):
        """Instant startup: loads dataset and fits TF-IDF in < 10ms without blocking network calls."""
        if isinstance(file_source, (str, Path)):
            p = str(file_source)
            self.df = pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
        else:
            try:
                self.df = pd.read_excel(file_source)
            except Exception:
                file_source.seek(0)
                self.df = pd.read_csv(file_source)

        self.df = self.df.dropna(how="all").fillna("")
        self.df = self.df.drop_duplicates()
        
        texts = []
        for _, row in self.df.iterrows():
            parts = []
            for col in self.df.columns:
                val = str(row[col]).strip()
                if val and not col.startswith("Unnamed"):
                    parts.append(f"{col}: {val}")
            texts.append(" | ".join(parts))
            
        self.df["__text__"] = texts
        # Fast in-memory indexing
        self.embeddings = self.tfidf.fit_transform(texts).toarray()
        return self.df

    def embed_company(self, company_details: dict, scraped_text: str = "") -> dict:
        company_name = company_details.get("company_name", "Target Enterprise")
        industry = company_details.get("industry_focus", "Enterprise Services")
        summary = company_details.get("corporate_summary", "")
        
        prev_proj = " ".join([p.get("project_title", "") + " " + p.get("description", "") for p in company_details.get("previous_projects", [])])
        curr_proj = " ".join([p.get("project_title", "") + " " + p.get("description", "") for p in company_details.get("current_projects", [])])
        needs = " ".join(company_details.get("expectations_and_needs", []))
        needs_sum = company_details.get("needs_summary", "")

        company_embedding_text = (
            f"Company: {company_name}. "
            f"Industry Focus: {industry}. "
            f"Corporate Overview: {summary}. "
            f"Delivered Operations & Projects: {prev_proj} {curr_proj}. "
            f"Strategic Requirements & Investment Scope: {needs}. {needs_sum}"
        ).strip()

        vector = self._get_single_worker_embedding(company_embedding_text, self.embedding_model_name)
        if vector is not None and len(vector) > 0:
            model_used = self.embedding_model_name
            tfidf_vec = self.tfidf.transform([company_embedding_text]).toarray()[0]
        else:
            tfidf_vec = self.tfidf.transform([company_embedding_text]).toarray()[0]
            vector = tfidf_vec
            model_used = "TF-IDF (Fast Direct Mode)"

        return {
            "embedding_text": company_embedding_text,
            "vector": vector,
            "tfidf_vector": tfidf_vec,
            "dimension": len(vector),
            "model_name": model_used,
            "vector_preview": [round(float(x), 4) for x in vector[:8]]
        }

    def match_company_vector(self, company_vector: np.ndarray, top_k: int = 3):
        if self.df is None or len(self.df) == 0:
            return []

        query_vec = company_vector
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        # If dimension mismatch (e.g. Worker AI 1024-dim vs TFIDF array), match using TF-IDF projection
        if query_vec.shape[1] != self.embeddings.shape[1]:
            if hasattr(self, "_last_tfidf_vec") and self._last_tfidf_vec is not None:
                query_vec = self._last_tfidf_vec.reshape(1, -1)
            else:
                return []

        sims = cosine_similarity(query_vec, self.embeddings)[0]
        sorted_indices = np.argsort(sims)[::-1]

        results = []
        seen_titles = set()

        for idx in sorted_indices:
            score = float(sims[idx])
            row = self.df.iloc[idx].to_dict()
            sector_name = str(row.get("Primary Sector") or row.get("Service Name", "")).strip()
            
            norm_title = re.sub(r"[^a-zA-Z0-9]", "", sector_name.lower())
            if norm_title in seen_titles:
                continue

            seen_titles.add(norm_title)
            row.pop("__text__", None)
            row["similarity"] = round(score, 3)
            row["match_pct"] = round(min(score * 130 + 40, 98.0), 1) if score > 0.05 else round(score * 100, 1)
            results.append(row)

            if len(results) >= top_k:
                break

        return results

catalog = ServiceCatalog()
