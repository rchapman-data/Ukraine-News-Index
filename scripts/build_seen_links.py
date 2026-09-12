"""
One-time setup step for Part 6: build a lightweight "seen links" manifest
from the existing cleaned corpus, so the RSS ingestion script
(scripts/ingest_rss.py) can check "have we already got this article"
without loading the full ~1.1 GB news_cleaned.csv on every scheduled run.

Only the `link` column is read from the CSV (via `usecols`), so this stays
fast and light on memory despite the source file's size.

Safe to rerun, but note it OVERWRITES the manifest from news_cleaned.csv
alone -- it does not merge in links ingest_rss.py has since added from RSS.
Only rerun this if you need to rebuild from scratch (e.g. after re-cleaning
the base corpus); day to day, leave it alone once created, since
ingest_rss.py appends newly-ingested links to it directly.
"""

from pathlib import Path

import pandas as pd

CLEANED_CSV_PATH = Path("data/processed/news_cleaned.csv")
SEEN_LINKS_PATH = Path("data/processed/seen_links.txt")


def main():
    print(f"Loading link column from {CLEANED_CSV_PATH} ...")
    df = pd.read_csv(CLEANED_CSV_PATH, usecols=["link"])
    print(f"Loaded {len(df):,} links")

    SEEN_LINKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEEN_LINKS_PATH, "w", encoding="utf-8") as f:
        for link in df["link"]:
            f.write(f"{link}\n")

    print(f"Saved {len(df):,} links to {SEEN_LINKS_PATH}")


if __name__ == "__main__":
    main()
