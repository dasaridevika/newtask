import os
import re
import urllib.parse
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

def clean_html_fast(raw_html: str) -> str:
    """Fast dependency-free text cleaner with regex (Zero BeautifulSoup)."""
    text = re.sub(r"<(script|style|nav|header|footer|svg|noscript)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()

def extract_links_fast(raw_html: str, base_url: str) -> list:
    parsed_base = urllib.parse.urlparse(base_url)
    domain = parsed_base.netloc
    
    matches = re.findall(r'href=["\'](.*?)["\']', raw_html, re.I)
    valid_links = []
    seen = set()

    priority_keywords = ["about", "product", "solution", "service", "case-stud", "portfolio", "investment", "investor", "infrastructure", "company", "overview", "news", "esg"]

    for m in matches:
        m = m.strip()
        if not m or m.startswith("#") or m.startswith("mailto:") or m.startswith("tel:") or m.startswith("javascript:"):
            continue
        full_url = urllib.parse.urljoin(base_url, m).split("#")[0].split("?")[0].rstrip("/")
        parsed_m = urllib.parse.urlparse(full_url)

        if parsed_m.netloc == domain and full_url not in seen:
            seen.add(full_url)
            if any(k in full_url.lower() for k in priority_keywords):
                valid_links.append(full_url)
            elif len(valid_links) < 10:
                valid_links.append(full_url)

    return valid_links[:10]

def fetch_page(url: str, timeout: int = 6) -> tuple:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200 and len(resp.text) > 200:
            return url, resp.text
    except Exception:
        pass
    return url, ""

def fetch_search_intelligence(company_name: str, domain: str) -> list:
    """Queries live search engines for high-depth corporate intelligence."""
    snippets = []
    queries = [
        f'"{company_name}" business model operations overview',
        f'"{company_name}" portfolio companies case studies projects',
        f'"{company_name}" capital investments manufacturing expansion news'
    ]

    for q in queries:
        try:
            duck_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1"
            resp = requests.get(duck_url, headers=HEADERS, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("AbstractText", "")
                heading = data.get("Heading", "")
                if abstract and len(abstract) > 50:
                    snippets.append(f"Entity Profile ({heading}): {abstract}")
                for topic in data.get("RelatedTopics", [])[:3]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        snippets.append(f"Topic Fact: {topic.get('Text')}")
        except Exception:
            pass

    return snippets

def search_company_serp(query_or_url: str, api_key: str = None) -> dict:
    """Comprehensive Dual-Engine Search & Scraping Engine."""
    if not query_or_url.startswith("http"):
        url = "https://" + query_or_url
    else:
        url = query_or_url

    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.replace("www.", "").strip()
    clean_name = domain.split(".")[0].capitalize()
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    all_sections = []
    source_links = [url]

    # 1. Fetch Landing Page
    _, home_html = fetch_page(url, timeout=6)
    
    if home_html:
        home_text = clean_html_fast(home_html)
        if len(home_text) > 80:
            all_sections.append(f"=== OFFICIAL HOMEPAGE ({url}) ===\n{home_text[:6000]}")

        sub_links = extract_links_fast(home_html, base_url)
        if sub_links:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(fetch_page, u, 5): u for u in sub_links}
                for future in as_completed(future_to_url):
                    sub_url, sub_html = future.result()
                    if sub_html:
                        sub_text = clean_html_fast(sub_html)
                        if len(sub_text) > 150:
                            all_sections.append(f"\n=== PAGE: {sub_url} ===\n{sub_text[:3500]}")
                            source_links.append(sub_url)

    # 2. Live Web Search Augmentation
    search_snippets = fetch_search_intelligence(clean_name, domain)
    if search_snippets:
        all_sections.append("\n=== LIVE VERIFIED SEARCH INTELLIGENCE ===\n" + "\n".join(search_snippets))

    # 3. Google SERP API (if key available)
    serp_key = api_key or os.getenv("SERPAPI_API_KEY") or os.getenv("SERP_API_KEY") or ""
    if serp_key:
        try:
            params = {"engine": "google", "q": f'"{clean_name}" corporate business overview past projects', "api_key": serp_key, "num": 5}
            resp = requests.get("https://serpapi.com/search", params=params, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                serp_res = [f"Search Result: {r.get('title')} - {r.get('snippet')}" for r in data.get("organic_results", []) if r.get("snippet")]
                if serp_res:
                    all_sections.append("\n=== GOOGLE SERP SEARCH INSIGHTS ===\n" + "\n".join(serp_res))
        except Exception:
            pass

    full_content = f"Company Domain: {domain}\nTarget Entity: {clean_name}\n\n" + "\n\n".join(all_sections)

    return {
        "domain": domain,
        "content": full_content,
        "search_results_count": max(len(all_sections), 1),
        "source_links": list(set(source_links))[:8],
        "categorized_summary": {
            "crawled_pages": len(all_sections),
            "verified_links": len(set(source_links))
        }
    }
