import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1"
}

class PageType(str, Enum):
    HOME = "home"
    ABOUT = "about"
    PRODUCTS_SERVICES = "products_services"
    SOLUTIONS = "solutions"
    CASE_STUDY = "case_study"
    NEWS_PRESS = "news_press"
    CAREERS = "careers"
    LEADERSHIP = "leadership"
    CONTACT = "contact"
    LEGAL = "legal"
    OTHER = "other"

@dataclass
class PageEvidence:
    url: str
    title: str
    page_type: PageType
    headings: List[str]
    clean_text: str
    canonical_snippets: List[str]
    extracted_keywords: List[str]
    credibility_weight: float
    timestamp: str
    status_code: int = 200

@dataclass
class BusinessSignal:
    category: str  # e.g., "products", "industry", "technology", "client", "expansion", "pain_point"
    signal: str
    source_url: str
    confidence: str  # "high", "medium", "low"
    snippet: str

@dataclass
class EvidenceStore:
    domain: str
    company_name: str
    base_url: str
    pages: List[PageEvidence] = field(default_factory=list)
    signals: List[BusinessSignal] = field(default_factory=list)
    observed_industries: List[str] = field(default_factory=list)
    observed_products: List[str] = field(default_factory=list)
    observed_technologies: List[str] = field(default_factory=list)
    observed_geographies: List[str] = field(default_factory=list)
    search_insights: List[str] = field(default_factory=list)
    confidence_score: float = 0.85
    confidence_label: str = "high"

    def get_aggregated_text(self, max_chars: int = 12000) -> str:
        """Assembles structured, weighted text from all evidence pages."""
        sections = []
        # Sort pages by credibility weight descending
        sorted_pages = sorted(self.pages, key=lambda p: p.credibility_weight, reverse=True)
        for page in sorted_pages:
            headings_str = " | ".join(page.headings[:5]) if page.headings else "N/A"
            sections.append(
                f"=== [{page.page_type.upper()}] {page.title} ({page.url}) ===\n"
                f"Headings: {headings_str}\n"
                f"Content Summary: {page.clean_text[:2500]}\n"
                f"Key Snippets: {' // '.join(page.canonical_snippets[:3])}\n"
            )
        if self.search_insights:
            sections.append("=== THIRD-PARTY VERIFIED SEARCH INSIGHTS ===\n" + "\n".join(self.search_insights[:5]))
        full = f"COMPANY: {self.company_name} (Domain: {self.domain})\n\n" + "\n".join(sections)
        return full[:max_chars]

    def to_dict(self) -> dict:
        return asdict(self)

def classify_page(url: str, title: str, headings: List[str], text_sample: str) -> tuple:
    """Classifies a page into PageType with credibility weighting."""
    path = urllib.parse.urlparse(url).path.lower().strip("/")
    combined_meta = f"{path} {title} {' '.join(headings)}".lower()

    if not path or path in ["", "index.html", "index.php", "home"]:
        return PageType.HOME, 1.0

    if any(k in combined_meta for k in ["product", "offering", "equipment", "hardware", "modules", "systems", "solutions", "service", "capabilities"]):
        return PageType.PRODUCTS_SERVICES, 1.4

    if any(k in combined_meta for k in ["case-stud", "case_study", "projects", "portfolio", "success-stories", "customers", "installations"]):
        return PageType.CASE_STUDY, 1.3

    if any(k in combined_meta for k in ["about", "company", "who-we-are", "history", "mission", "overview"]):
        return PageType.ABOUT, 1.1

    if any(k in combined_meta for k in ["press", "news", "announcement", "media", "blog", "insights"]):
        return PageType.NEWS_PRESS, 0.9

    if any(k in combined_meta for k in ["career", "jobs", "join-us", "hiring", "openings"]):
        return PageType.CAREERS, 0.8

    if any(k in combined_meta for k in ["leader", "board", "team", "executive", "management"]):
        return PageType.LEADERSHIP, 0.9

    if any(k in combined_meta for k in ["contact", "get-in-touch", "locations", "offices"]):
        return PageType.CONTACT, 0.7

    if any(k in combined_meta for k in ["privacy", "terms", "legal", "cookie", "disclaimer", "accessibility"]):
        return PageType.LEGAL, 0.2

    return PageType.OTHER, 0.6

def clean_html(raw_html: str) -> dict:
    """Extracts structured metadata, title, headings, clean text, and canonical snippets."""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.DOTALL)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else "Enterprise Page"
    title = title.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")

    headings = []
    for h in re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", raw_html, re.I | re.DOTALL):
        clean_h = re.sub(r"<[^>]+>", "", h).strip()
        clean_h = re.sub(r"\s+", " ", clean_h)
        if 4 < len(clean_h) < 120 and clean_h not in headings:
            headings.append(clean_h)

    # Strip scripts, styles, navigation, footer, svgs
    text = re.sub(r"<(script|style|nav|header|footer|svg|noscript|iframe)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Extract high-value sentences for canonical snippets
    sentences = re.split(r"(?<=[.!?])\s+", text)
    canonical_snippets = []
    signal_keywords = [
        "manufactur", "develop", "provide", "specialize", "leader", "operat", "solution",
        "power", "solar", "photovoltaic", "thermal", "cooling", "data center", "investment",
        "private equity", "fund", "portfolio", "acquir", "infrastructure", "capacity", "megawatt",
        "gigawatt", "clean energy", "patent", "facility", "plant", "oem", "supply"
    ]

    for sent in sentences:
        clean_s = sent.strip()
        if 40 < len(clean_s) < 260:
            if any(k in clean_s.lower() for k in signal_keywords):
                if clean_s not in canonical_snippets:
                    canonical_snippets.append(clean_s)
                    if len(canonical_snippets) >= 6:
                        break

    return {
        "title": title,
        "headings": headings[:8],
        "clean_text": text,
        "canonical_snippets": canonical_snippets
    }

def extract_links(raw_html: str, base_url: str) -> List[str]:
    """Finds verified subpage URLs belonging to the same host."""
    parsed_base = urllib.parse.urlparse(base_url)
    domain = parsed_base.netloc
    
    matches = re.findall(r'href=["\'](.*?)["\']', raw_html, re.I)
    priority_links = []
    seen = set()

    priority_keywords = [
        "product", "solution", "service", "case-stud", "portfolio", "technology", "tech",
        "solar", "module", "system", "infrastructure", "thermal", "cooling", "power", 
        "manufacturing", "about", "company", "projects", "press", "news"
    ]

    for m in matches:
        m = m.strip()
        if not m or m.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        full_url = urllib.parse.urljoin(base_url, m).split("#")[0].split("?")[0].rstrip("/")
        parsed_m = urllib.parse.urlparse(full_url)

        if (parsed_m.netloc == domain or parsed_m.netloc == f"www.{domain}" or domain in parsed_m.netloc) and full_url not in seen:
            seen.add(full_url)
            if any(k in full_url.lower() for k in priority_keywords) and full_url != base_url:
                priority_links.append(full_url)

    return priority_links[:10]

def fetch_page_content(url: str, timeout: int = 7) -> tuple:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200 and len(resp.text) > 200:
            return url, resp.text, resp.status_code
    except Exception:
        pass
    return url, "", 500

def extract_deterministic_signals(text: str, url: str) -> List[BusinessSignal]:
    """Deterministic signal extraction for domain-specific industrial categories."""
    signals = []
    lower = text.lower()

    # 1. Solar / Renewable Energy Signals
    solar_matches = re.findall(r"\b(solar (?:module|panel|cell|power|farm|energy|inverter)|photovoltaic|bifacial|perovskite|pv module|utility-scale solar|distributed generation|bess|battery storage)\b", lower)
    if solar_matches:
        unique_matches = list(set(solar_matches))
        signals.append(BusinessSignal(
            category="products_and_capabilities",
            signal=f"Solar & Clean Energy Manufacturing: {', '.join(unique_matches[:4])}",
            source_url=url,
            confidence="high",
            snippet=f"Detected specialized clean energy terms: {', '.join(unique_matches)}"
        ))

    # 2. Mission-Critical Thermal / Power OEM Signals
    thermal_matches = re.findall(r"\b(thermal management|liquid cooling|direct-to-chip|cdu|chiller|switchgear|uninterruptible power supply|ups system|modular data center|busway|power distribution)\b", lower)
    if thermal_matches:
        unique_matches = list(set(thermal_matches))
        signals.append(BusinessSignal(
            category="products_and_capabilities",
            signal=f"Mission-Critical Power & Thermal Systems: {', '.join(unique_matches[:4])}",
            source_url=url,
            confidence="high",
            snippet=f"Detected critical infrastructure terms: {', '.join(unique_matches)}"
        ))

    # 3. Private Equity / Buyout / Asset Management Signals
    pe_matches = re.findall(r"\b(assets under management|aum|middle market|private equity|buyout|private debt|growth equity|portfolio companies|investment strategy|add-on acquisition)\b", lower)
    if pe_matches:
        unique_matches = list(set(pe_matches))
        signals.append(BusinessSignal(
            category="business_model",
            signal=f"Institutional Private Equity & Asset Management: {', '.join(unique_matches[:4])}",
            source_url=url,
            confidence="high",
            snippet=f"Detected private equity investment strategy: {', '.join(unique_matches)}"
        ))

    # 4. Expansion / Growth / Construction Signals
    expansion_matches = re.findall(r"\b(new facility|manufacturing plant|gigawatt factory|capacity expansion|commissioned|breaking ground|capital expenditure|capex|mw capacity)\b", lower)
    if expansion_matches:
        unique_matches = list(set(expansion_matches))
        signals.append(BusinessSignal(
            category="growth_and_capex",
            signal=f"Capital Expansion & Infrastructure Buildout: {', '.join(unique_matches[:3])}",
            source_url=url,
            confidence="high",
            snippet=f"Detected physical asset scaling: {', '.join(unique_matches)}"
        ))

    return signals

def fetch_search_insights(company_name: str, domain: str) -> List[str]:
    """Retrieves verified third-party search intelligence snippets."""
    insights = []
    queries = [
        f'"{company_name}" operations products technology overview',
        f'"{company_name}" manufacturing facilities capacity plants',
        f'"{company_name}" industry focus business model'
    ]

    for q in queries:
        try:
            duck_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1"
            resp = requests.get(duck_url, headers=HEADERS, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("AbstractText", "")
                heading = data.get("Heading", "")
                if abstract and len(abstract) > 40:
                    insights.append(f"Search Dossier ({heading}): {abstract}")
                for topic in data.get("RelatedTopics", [])[:2]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        insights.append(f"Fact: {topic.get('Text')}")
        except Exception:
            pass

    return insights

def search_company_serp(query_or_url: str, api_key: str = None) -> dict:
    """
    Main entry point: crawls target domain, classifies pages into structured PageEvidence objects,
    extracts business signals, and returns a verified EvidenceStore.
    """
    if not query_or_url.startswith("http"):
        url = "https://" + query_or_url
    else:
        url = query_or_url

    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.replace("www.", "").strip()
    clean_name = domain.split(".")[0].capitalize()
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    store = EvidenceStore(
        domain=domain,
        company_name=clean_name,
        base_url=base_url
    )

    # 1. Direct Landing Page Ingestion
    _, home_html, status_code = fetch_page_content(url, timeout=7)
    if home_html:
        extracted = clean_html(home_html)
        p_type, weight = classify_page(url, extracted["title"], extracted["headings"], extracted["clean_text"])
        
        home_evidence = PageEvidence(
            url=url,
            title=extracted["title"],
            page_type=p_type,
            headings=extracted["headings"],
            clean_text=extracted["clean_text"],
            canonical_snippets=extracted["canonical_snippets"],
            extracted_keywords=[],
            credibility_weight=weight,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status_code=status_code
        )
        store.pages.append(home_evidence)
        store.signals.extend(extract_deterministic_signals(extracted["clean_text"], url))

        # 2. Parallel Subpage Ingestion
        sub_links = extract_links(home_html, base_url)
        if sub_links:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(fetch_page_content, u, 6): u for u in sub_links}
                for future in as_completed(future_to_url):
                    sub_url, sub_html, sub_status = future.result()
                    if sub_html and sub_status == 200:
                        sub_ext = clean_html(sub_html)
                        sub_type, sub_weight = classify_page(sub_url, sub_ext["title"], sub_ext["headings"], sub_ext["clean_text"])
                        if len(sub_ext["clean_text"]) > 100:
                            evidence = PageEvidence(
                                url=sub_url,
                                title=sub_ext["title"],
                                page_type=sub_type,
                                headings=sub_ext["headings"],
                                clean_text=sub_ext["clean_text"],
                                canonical_snippets=sub_ext["canonical_snippets"],
                                extracted_keywords=[],
                                credibility_weight=sub_weight,
                                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                status_code=sub_status
                            )
                            store.pages.append(evidence)
                            store.signals.extend(extract_deterministic_signals(sub_ext["clean_text"], sub_url))

    # 3. Third-party Search Verification
    search_insights = fetch_search_insights(clean_name, domain)
    store.search_insights = search_insights

    # 4. Confidence Rubric Calculation
    total_pages = len(store.pages)
    types_found = {p.page_type for p in store.pages}
    has_product_page = PageType.PRODUCTS_SERVICES in types_found or PageType.SOLUTIONS in types_found
    
    if total_pages >= 4 and has_product_page and len(store.signals) >= 2:
        store.confidence_score = 0.94
        store.confidence_label = "high"
    elif total_pages >= 2:
        store.confidence_score = 0.82
        store.confidence_label = "medium"
    else:
        store.confidence_score = 0.65
        store.confidence_label = "low"

    # Assemble backward-compatible dictionary payload for downstream consumers
    aggregated_text = store.get_aggregated_text()

    return {
        "domain": domain,
        "base_url": base_url,
        "content": aggregated_text,
        "evidence_store": store,
        "search_results_count": len(store.pages),
        "source_links": [p.url for p in store.pages]
    }
