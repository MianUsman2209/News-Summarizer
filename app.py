%%writefile extractor.py
import asyncio
import json
import re
from typing import Optional, Dict, Any, List

import requests
from bs4 import BeautifulSoup

try:
    from crawl4ai import AsyncWebCrawler
    HAVE_CRAWL4AI = True
except Exception:
    HAVE_CRAWL4AI = False

from transformers import pipeline

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

print("Loading summarizer model... (may take 20-60s on first run)")
summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device=0 if _import_("torch").cuda.is_available() else -1)
print("Summarizer ready.")
#3
def safe_requests_get(url: str, timeout: int = 15) -> Optional[str]:
    """Robust GET with headers."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ColabBot/1.0; +https://example.com/bot)"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[requests] Error fetching {url}: {e}")
        return None

async def fetch_with_crawl4ai(url: str) -> Optional[str]:
    """Attempt to fetch page HTML using crawl4ai AsyncWebCrawler (if available)."""
    if not HAVE_CRAWL4AI:
        return None
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

        if not result or not getattr(result, "success", True):
            return None

        # Prefer HorTML  text
        html = None
        if hasattr(result, "html"):
            html = result.html
        if html:
            return html
        if hasattr(result, "text"):
            return result.text
        return None
    except Exception as e:
        print(f"[crawl4ai] Error: {e}")
        return None



def parse_json_ld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Return list of parsed JSON-LD blocks (if any)."""
    items = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "{}")
            items.append(data)
        except Exception:
            try:
                text = tag.string or ""
                # quick cleanup
                text = re.sub(r"\s+", " ", text)
                data = json.loads(text)
                items.append(data)
            except Exception:
                continue
    return items

def extract_meta(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract metadata from HTML meta tags and common patterns."""
    meta = {}
    # title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    # meta tags
    def _(name):
        t = soup.find("meta", attrs={"name": name})
        if t and t.get("content"):
            return t["content"].strip()
        t = soup.find("meta", attrs={"property": name})
        if t and t.get("content"):
            return t["content"].strip()
        return None

    og_title = _("og:title")
    description = _("description") or _("og:description") or _("twitter:description")
    author = _("author") or _("article:author") or _("og:article:author")
    pub_time = _("article:published_time") or _("og:published_time") or _("publication_date") or _("date")
    section = _("article:section") or _("og:section")
    keywords = _("keywords") or _("news_keywords") or _("og:tags")
    if og_title:
        title = og_title

    meta.update({
        "title": title,
        "description": description,
        "author": author,
        "published_time": pub_time,
        "section": section,
        "keywords": keywords
    })
    return meta

def extract_main_text(soup: BeautifulSoup) -> str:
    """Heuristic: try <article>, fallback to largest <div> with many <p>."""
    # 1) article tag
    article = soup.find("article")
    if article:
        paragraphs = article.find_all("p")
        if paragraphs:
            return "\n\n".join([p.get_text(separator=" ", strip=True) for p in paragraphs if p.get_text(strip=True)])
        # fallback to article text
        return article.get_text(separator="\n", strip=True)

    # 2) look for main tag
    main = soup.find("main")
    if main:
        paras = main.find_all("p")
        if paras:
            return "\n\n".join([p.get_text(separator=" ", strip=True) for p in paras if p.get_text(strip=True)])

    # 3) heuristic: largest block of consecutive <p> inside a div
    best = ""
    for div in soup.find_all(["div", "section"], recursive=True):
        paras = div.find_all("p")
        text = "\n\n".join([p.get_text(separator=" ", strip=True) for p in paras if p.get_text(strip=True)])
        if len(text) > len(best):
            best = text
    # 4) as last resort, combine all <p>
    if best:
        return best
    all_p = soup.find_all("p")
    return "\n\n".join([p.get_text(separator=" ", strip=True) for p in all_p if p.get_text(strip=True)])

def chunk_text_by_chars(text: str, max_chars: int = 3000) -> List[str]:
    """Simple character-based chunker to avoid token-lib dependencies."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + max_chars
        # try to break at sentence boundary
        if end < n:
            idx = text.rfind(".", start, end)
            if idx != -1 and idx - start > max_chars // 4:
                end = idx + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks

def safe_summarize(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """Use HF summarizer pipeline with limited size; fall back to truncation if it fails."""
    text = text.strip()
    if not text:
        return ""
    try:
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]["summary_text"].strip()
    except Exception as e:
        # fallback: take first n chars + ellipsis
        if len(text) <= max_length * 6:
            return text[:max_length*2].strip() + "..."
        return text[: max_length * 2].strip() + "..."

def choose_category_by_keywords(text: str) -> str:
    """Very simple category classifier by keywords."""
    t = text.lower()
    categories = {
        "politics": ["president", "government", "election", "minister", "congress", "parliament"],
        "sports": ["match", "tournament", "score", "goal", "season", "coach", "player"],
        "tech": ["technology", "tech", "software", "ai ", "artificial intelligence", "startup"],
        "business": ["market", "stock", "company", "finance", "economy", "investor"],
        "health": ["health", "disease", "vaccine", "hospital", "covid", "mental"],
        "science": ["research", "study", "scientists", "physics", "space", "experiment"],
        "entertainment": ["film", "movie", "music", "celebrity", "tv", "show"]
    }
    for cat, kws in categories.items():
        for kw in kws:
            if kw in t:
                return cat.capitalize()
    return "General"
# 4Main extraction function


async def _fetch_html(url: str) -> Optional[str]:
    """Try crawl4ai first (async), else requests fallback."""
    if HAVE_CRAWL4AI:
        html = await fetch_with_crawl4ai(url)
        if html:
            return html
    # fallback to requests
    return safe_requests_get(url)

async def crawl_and_extract(url: str) -> Dict[str, Any]:
    """Fetch, parse, chunk/summarize and return structured fields."""
    data = {
        "url": url,
        "Headline": None,
        "Body": None,
        "Author": None,
        "Publication date & time": None,
        "Category": None,
        "Tags": [],
        "Excerpt": None
    }

    html = await _fetch_html(url)
    if not html:
        print(f"Failed to fetch content for {url}")
        return data

    soup = BeautifulSoup(html, "lxml")

    meta = extract_meta(soup)
    json_ld = parse_json_ld(soup)

    if json_ld:
        for item in json_ld:
            if not isinstance(item, dict):
                continue
            # If item is an array under '@graph', iterate
            items = [item]
            if "@graph" in item and isinstance(item["@graph"], list):
                items = item["@graph"]
            for node in items:
                # Many sites use "@type": "NewsArticle" or "Article"
                t = node.get("@type") or node.get("type") or ""
                if isinstance(t, list):
                    t = t[0] if t else ""
                if isinstance(t, str) and t.lower() in ("newsarticle", "article", "blogposting"):
                    # title
                    if not data["Headline"]:
                        data["Headline"] = node.get("headline") or node.get("name") or data["Headline"]
                    # author
                    a = node.get("author")
                    if a:
                        if isinstance(a, dict):
                            data["Author"] = a.get("name") or data["Author"]
                        elif isinstance(a, list) and a:
                            if isinstance(a[0], dict):
                                data["Author"] = a[0].get("name") or data["Author"]
                            else:
                                data["Author"] = str(a[0])
                        else:
                            data["Author"] = str(a)
                    # datePublished
                    if node.get("datePublished"):
                        data["Publication date & time"] = node.get("datePublished")
                    # articleSection
                    if node.get("articleSection"):
                        data["Category"] = node.get("articleSection")
                    # keywords
                    kws = node.get("keywords")
                    if kws and not data["Tags"]:
                        if isinstance(kws, list):
                            data["Tags"] = kws
                        else:
                            data["Tags"] = [k.strip() for k in str(kws).split(",") if k.strip()]

    # 2) fallback meta tags
    if not data["Headline"]:
        data["Headline"] = meta.get("title") or (soup.title.string.strip() if soup.title and soup.title.string else None)
    if not data["Author"]:
        data["Author"] = meta.get("author")
    if not data["Publication date & time"]:
        data["Publication date & time"] = meta.get("published_time")
    if not data["Category"]:
        data["Category"] = meta.get("section")
    if not data["Tags"] and meta.get("keywords"):
       data["Tags"] = [k.strip() for k in meta["keywords"].split(",") if k.strip()]

    body_text = extract_main_text(soup)
    data["Body"] = body_text or None

    if data["Body"]:
        chunks = chunk_text_by_chars(data["Body"], max_chars=3000)
        if len(chunks) == 1:
            excerpt = safe_summarize(data["Body"], max_length=130, min_length=30)
        else:
            summaries = []
            for ch in chunks:
                summaries.append(safe_summarize(ch, max_length=120, min_length=20))
            combined = "\n\n".join(summaries)
            excerpt = safe_summarize(combined, max_length=150, min_length=40)
        data["Excerpt"] = excerpt
    if not data["Category"] or data["Category"] == "":
        data["Category"] = choose_category_by_keywords((data["Body"] or "")[:4000])

    if not data["Tags"]:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", (data["Body"] or "").lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:6]
        data["Tags"] = [w for w, _ in top] if top else ["general"]

    for k in ["Headline", "Author", "Publication date & time", "Category", "Excerpt", "Body"]:
        if data.get(k) and isinstance(data[k], str):
            data[k] = data[k].strip()

    return data
