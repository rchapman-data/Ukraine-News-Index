"""
Fetch Ukrainska Pravda's Ukrainian-language "News" RSS feed, keep only
genuinely new articles (not already in the corpus), and save them to their
own dated file under data/incremental/ -- the actual Part 6 ingestion logic,
building on the parsing/cleaning already validated in scripts/fetch_rss.py.

Feed and cleaning approach match fetch_rss.py exactly (see that script's
docstring for the full reasoning on feed choice, the empty-content quirk,
and the HTML cleanup rules). What's new here:

1. Dedup against the corpus, not just within one feed pull. Checks each
   kept item's `link` against data/processed/seen_links.txt (built once by
   scripts/build_seen_links.py from news_cleaned.csv, then kept up to date
   by this script). This avoids loading the full ~1.1 GB base CSV just to
   check membership.

2. Genuinely new articles are saved to their own file --
   data/incremental/<UTC timestamp>.csv -- rather than rewriting the base
   CSV. Anything that later needs the full corpus (search, benchmarking)
   loads the base file plus all incremental shards. Columns match the base
   corpus's schema; title_ukrainian/text_ukrainian are left blank, since
   this Ukrainian-language feed has no separate translated version to
   populate them with (only the historical corpus's dual-language rows do).

3. Newly-saved links are appended to seen_links.txt immediately, so a
   second run against an unchanged feed reports everything as already seen
   -- this is the actual dedup behaviour to check when testing.

Run this locally and read the printed summary. No S3/AWS involved yet.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import pandas as pd
from bs4 import BeautifulSoup

RSS_FEED_URL = "https://www.pravda.com.ua/rss/view_news/"

SEEN_LINKS_PATH = Path("data/processed/seen_links.txt")
INCREMENTAL_DIR = Path("data/incremental")

# Below this many characters, treat the article as "not available yet" and
# skip it -- it will be checked again on the next run since it's never
# added to seen_links.txt.
MIN_CONTENT_CHARS = 50

BOILERPLATE_MARKERS = ["Patreon", "patreon.com"]

CORPUS_COLUMNS = [
    "link", "date", "title_original", "title_ukrainian",
    "text_original", "text_ukrainian", "category",
]


def clean_html_to_text(html_fragment):
    soup = BeautifulSoup(html_fragment, "html.parser")
    for template_tag in soup.find_all("template"):
        template_tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def strip_trailing_boilerplate(text):
    lowered = text.lower()
    cut_points = [
        lowered.find(marker.lower())
        for marker in BOILERPLATE_MARKERS
        if marker.lower() in lowered
    ]
    if cut_points:
        text = text[: min(cut_points)].rstrip()
    return text


def load_seen_links(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/build_seen_links.py first"
        )
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main():
    seen_links = load_seen_links(SEEN_LINKS_PATH)
    print(f"Loaded {len(seen_links):,} known links from {SEEN_LINKS_PATH}")

    print(f"Fetching {RSS_FEED_URL} ...")
    feed = feedparser.parse(RSS_FEED_URL)
    print(f"Found {len(feed.entries)} items\n")

    new_rows = []
    skipped_no_content = 0
    skipped_already_seen = 0

    for entry in feed.entries:
        raw_html = ""
        if "content" in entry and len(entry.content) > 0:
            raw_html = entry.content[0].value

        text = clean_html_to_text(raw_html)
        text = strip_trailing_boilerplate(text)

        if len(text) < MIN_CONTENT_CHARS:
            skipped_no_content += 1
            print(f"[SKIP - no content yet] {entry.title}")
            continue

        if entry.link in seen_links:
            skipped_already_seen += 1
            print(f"[SKIP - already have it] {entry.title}")
            continue

        date_str = ""
        if entry.get("published_parsed"):
            date_str = time.strftime("%d.%m.%Y", entry.published_parsed)

        new_rows.append({
            "link": entry.link,
            "date": date_str,
            "title_original": entry.title,
            "title_ukrainian": "",
            "text_original": text,
            "text_ukrainian": "",
            "category": entry.get("category", ""),
        })
        print(f"[NEW] {entry.title}")

    if new_rows:
        INCREMENTAL_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        output_path = INCREMENTAL_DIR / f"{timestamp}.csv"

        pd.DataFrame(new_rows, columns=CORPUS_COLUMNS).to_csv(output_path, index=False)
        print(f"\nSaved {len(new_rows)} new article(s) to {output_path}")

        with open(SEEN_LINKS_PATH, "a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(row["link"] + "\n")
        print(f"Appended {len(new_rows)} link(s) to {SEEN_LINKS_PATH}")
    else:
        print("\nNo new articles this run -- nothing to save.")

    print(
        f"\nSummary: {len(feed.entries)} fetched, {len(new_rows)} new, "
        f"{skipped_no_content} skipped (no content yet), "
        f"{skipped_already_seen} already seen"
    )


if __name__ == "__main__":
    main()
