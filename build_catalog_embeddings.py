import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "primary_sector_with_definitions.csv"
OUTPUT_NPZ = BASE_DIR / "catalog_embeddings.npz"

WORKER_URL = os.getenv("CLOUDFLARE_WORKER_URL", "https://lead-research-ai-worker.devika-worker.workers.dev")
MODEL_NAME = os.getenv("CF_EMBEDDING_MODEL", "@cf/baai/bge-large-en-v1.5")

def generate_embeddings():
    print(f"Loading catalog from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH).dropna(subset=["Primary Sector"]).fillna("")
    
    sectors = df["Primary Sector"].tolist()
    definitions = df["Definition"].tolist() if "Definition" in df.columns else [""] * len(sectors)
    
    texts = [f"Sector: {s}. Definition: {d}" for s, d in zip(sectors, definitions)]
    print(f"Total catalog sectors to embed: {len(texts)}")

    chunk_size = 25
    all_vectors = []

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        print(f"Embedding chunk {i // chunk_size + 1}/{(len(texts) + chunk_size - 1) // chunk_size} ({len(chunk)} items)...")
        
        success = False
        for attempt in range(3):
            try:
                resp = requests.post(
                    WORKER_URL.rstrip("/") + "/ai/embed",
                    json={"model": MODEL_NAME, "text": chunk},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    for item in data:
                        if isinstance(item, list):
                            all_vectors.append(item)
                        elif isinstance(item, dict) and "values" in item:
                            all_vectors.append(item["values"])
                    success = True
                    break
                else:
                    print(f"  Attempt {attempt+1} failed with status {resp.status_code}: {resp.text}")
                    time.sleep(1)
            except Exception as e:
                print(f"  Attempt {attempt+1} error: {e}")
                time.sleep(1)
        
        if not success:
            raise RuntimeError(f"Failed to embed chunk starting at index {i}")

    vectors_matrix = np.array(all_vectors, dtype=np.float32)
    # L2 Normalize for lightning-fast dot product cosine similarity
    norms = np.linalg.norm(vectors_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized_vectors = vectors_matrix / norms

    print(f"Successfully generated normalized vector matrix shape: {normalized_vectors.shape}")
    
    np.savez_compressed(
        OUTPUT_NPZ,
        vectors=normalized_vectors,
        sectors=np.array(sectors, dtype=object),
        definitions=np.array(definitions, dtype=object),
        texts=np.array(texts, dtype=object),
        model_name=MODEL_NAME
    )
    print(f"Saved catalog embeddings to {OUTPUT_NPZ} ({os.path.getsize(OUTPUT_NPZ) / 1024:.1f} KB)")

if __name__ == "__main__":
    generate_embeddings()
