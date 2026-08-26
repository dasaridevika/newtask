# Lead Research & Strategic Offering Matcher

Deep Multi-Page Website Intelligence Extraction → **1024-Dim Dense Vector Embedding** (`@cf/baai/bge-large-en-v1.5`) → Cosine Similarity Search → Senior Principal Executive Assessment & Requirement-to-Service Mapping.

---

## ⚡ Key Architecture & Features

1. **Dual-Engine Factual Ingestion (`scraper.py`)**:
   - Ultra-fast concurrent crawler (harvests 5–9 subpages in $\approx 3.5\text{s}$) with zero BeautifulSoup dependency.
   - Automated anti-bot search fallback for protected enterprise domains.

2. **1024-Dimensional Dense Vector Embedding (`service_catalog.py`)**:
   - Connected directly to Cloudflare Workers AI embedding endpoint (`/ai/embed`).
   - In-memory Cosine Similarity ranking against 462 primary sector definitions in $< 5\text{ms}$.

3. **Strategic Senior Principal Synthesis (`worker_ai.py`)**:
   - Deployed on Cloudflare Workers AI (`@cf/meta/llama-3.2-3b-instruct`).
   - Anti-placeholder sanitization that eliminates template bracket hallucinations.
   - Formats comprehensive research reports on **Delivered Projects**, **Active Operations**, and **Future Roadmaps** with direct source links.

4. **Grand Glassmorphic Executive Dashboard (`app.py`)**:
   - High-contrast obsidian slate theme with translucent frosted glass cards.
   - 5 structured tabs: Executive Dossier, Projects Research, Offering Mapping, ROI/Pitch, and Vector Inspector.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Dashboard
```bash
streamlit run app.py
```
