import argparse
import json
import re
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "research-agent-learning-project/0.1 (educational)"}

SEED_CATEGORIES = [
    "History of computing",
    "History of computing hardware",
    "History of software",
    "Computer pioneers",
    "History of the Internet",
]
MAX_DEPTH = 2

CHUNK_WORDS = 250
OVERLAP_WORDS = 40
MIN_ARTICLE_WORDS = 150

OUT = Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"


def api_get(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    last = None
    for attempt in range(6):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 5)) + attempt * 5)
                last = "429 rate limited"
                continue
            r.raise_for_status()
            time.sleep(0.5)
            return r.json()
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"MediaWiki API kept failing: {last}")


def category_members(cat: str):
    cont = {}
    while True:
        data = api_get(
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{cat}",
                "cmlimit": "500",
                "cmtype": "page|subcat",
                **cont,
            }
        )
        for m in data["query"]["categorymembers"]:
            if m["ns"] == 14:
                yield m["title"].removeprefix("Category:"), True
            elif m["ns"] == 0:
                yield m["title"], False
        if "continue" not in data:
            return
        cont = data["continue"]


def collect_titles(max_articles: int) -> list[str]:
    titles: dict[str, None] = {}
    seen_cats = set()
    frontier = [(c, 0) for c in SEED_CATEGORIES]
    while frontier and len(titles) < max_articles:
        cat, depth = frontier.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        for name, is_cat in category_members(cat):
            if is_cat:
                if depth < MAX_DEPTH:
                    frontier.append((name, depth + 1))
            else:
                titles.setdefault(name)
        print(f"  crawled '{cat}' (depth {depth}) -> {len(titles)} titles")
    return list(titles)[:max_articles]


def fetch_one(title: str) -> dict | None:
    # The API only returns FULL extracts for one page per request (batching needs exintro).
    data = api_get(
        {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": "1",
            "exsectionformat": "plain",
            "inprop": "url",
            "titles": title,
            "redirects": "1",
        }
    )
    for p in data["query"]["pages"]:
        text = p.get("extract") or ""
        if len(text.split()) >= MIN_ARTICLE_WORDS:
            return {"title": p["title"], "url": p["fullurl"], "text": text}
    return None


CACHE = OUT.parent / "articles_cache.jsonl"


def fetch_extracts(titles: list[str]) -> list[dict]:
    cached = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            a = json.loads(line)
            cached[a["query_title"]] = a
    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as f:
        for i, t in enumerate(titles, start=1):
            if t not in cached:
                a = fetch_one(t) or {"title": t, "url": "", "text": ""}
                a["query_title"] = t
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
                f.flush()
                cached[t] = a
            if i % 10 == 0:
                print(f"  {i}/{len(titles)}", flush=True)
    return [cached[t] for t in titles if cached[t]["text"]]


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def clean(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and (len(ln.split()) > 6 or ln[-1] in ".!?")]
    return " ".join(lines)


def chunk_text(text: str) -> list[str]:
    sentences = SENT_SPLIT.split(clean(text))
    chunks, cur, cur_words = [], [], 0
    for s in sentences:
        n = len(s.split())
        if cur and cur_words + n > CHUNK_WORDS:
            chunks.append(" ".join(cur))
            tail, w = [], 0
            for prev in reversed(cur):
                pw = len(prev.split())
                if w + pw > OVERLAP_WORDS and tail:
                    break
                tail.insert(0, prev)
                w += pw
            cur, cur_words = tail, w
        cur.append(s)
        cur_words += n
    if cur and (not chunks or cur_words > OVERLAP_WORDS + 20):
        chunks.append(" ".join(cur))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-articles", type=int, default=700)
    args = ap.parse_args()

    print("Collecting titles...")
    titles = collect_titles(args.max_articles)
    print(f"Fetching {len(titles)} articles...")
    articles = fetch_extracts(titles)

    OUT.parent.mkdir(exist_ok=True)
    next_id = 1
    with OUT.open("w", encoding="utf-8") as f:
        for a in articles:
            for c in chunk_text(a["text"]):
                f.write(
                    json.dumps(
                        {
                            "id": next_id,
                            "title": a["title"],
                            "url": a["url"],
                            "text": c,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                next_id += 1
    print(f"Wrote {next_id - 1} chunks from {len(articles)} articles to {OUT}")


if __name__ == "__main__":
    main()
