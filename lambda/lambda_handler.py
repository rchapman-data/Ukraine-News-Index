"""
Lambda handler for Part 6's scheduled RSS ingestion + embedding.

S3-based port of scripts/ingest_rss.py + scripts/embed_incremental.py --
same feed, same cleaning rules, same dedup approach, same embedding model
and settings -- the only real change is swapping every local file path for
an S3 read/write, since a Lambda has no persistent disk between runs (only
/tmp, and Lambda gives every invocation a fresh, empty /tmp -- nothing
written there in one run is visible in the next).

Combined into one handler (ingest, then embed) rather than two separate
Lambdas, since both steps are small and already run back-to-back locally --
see build-progress.md, Part 6, for the full reasoning behind every design
choice below.

S3 layout (bucket set via the S3_BUCKET env var):
    processed/seen_links.txt                -- dedup manifest, read + rewritten
    incremental/<timestamp>.csv             -- newly-ingested articles
    embeddings/incremental/<name>.npy       -- matching embeddings
    embeddings/incremental/<name>_meta.csv  -- matching metadata

Note on seen_links.txt: locally, ingest_rss.py appends new links to the
file in place. S3 objects can't be appended to, so this version downloads
the whole manifest, adds the new links in memory, and re-uploads the whole
thing. Functionally the same end state, just a different mechanism.

The embedding model itself is baked into the Docker image at build time
(see the Dockerfile) -- this handler only ever loads it from files already
inside the image, it never downloads anything from Hugging Face.
"""

import io
import os
import time
from datetime import datetime, timezone

import boto3
import feedparser
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

RSS_FEED_URL = "https://www.pravda.com.ua/rss/view_news/"

BUCKET_NAME = os.environ.get("S3_BUCKET", "rcchap79-ukraine-news-index")

SEEN_LINKS_KEY = "processed/seen_links.txt"
INCREMENTAL_PREFIX = "incremental/"
EMBEDDINGS_INCREMENTAL_PREFIX = "embeddings/incremental/"

# Below this many characters, treat the article as "not available yet" and
# skip it -- it will be checked again on the next scheduled run since it's
# never added to seen_links.txt. Same rule as fetch_rss.py/ingest_rss.py.
MIN_CONTENT_CHARS = 50

BOILERPLATE_MARKERS = ["Patreon", "patreon.com"]

CORPUS_COLUMNS = [
    "link", "date", "title_original", "title_ukrainian",
    "text_original", "text_ukrainian", "category",
]

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 128

s3 = boto3.client("s3")


# --- Cleaning helpers: identical to fetch_rss.py / ingest_rss.py ---------

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


# --- Ingestion: S3 port of ingest_rss.py ----------------------------------

def load_seen_links():
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=SEEN_LINKS_KEY)
    body = obj["Body"].read().decode("utf-8")
    return {line.strip() for line in body.splitlines() if line.strip()}


def save_seen_links(seen_links):
    body = "\n".join(sorted(seen_links)) + "\n"
    s3.put_object(Bucket=BUCKET_NAME, Key=SEEN_LINKS_KEY, Body=body.encode("utf-8"))


def fetch_new_articles(seen_links):
    """Same logic as ingest_rss.py's main() loop, returns rows + counts
    instead of printing them line by line."""
    feed = feedparser.parse(RSS_FEED_URL)

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
            continue

        if entry.link in seen_links:
            skipped_already_seen += 1
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

    return new_rows, len(feed.entries), skipped_no_content, skipped_already_seen


def save_incremental_csv(new_rows):
    df = pd.DataFrame(new_rows, columns=CORPUS_COLUMNS)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    key = f"{INCREMENTAL_PREFIX}{timestamp}.csv"

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buffer.getvalue().encode("utf-8"))
    return key


# --- Embedding: S3 port of embed_incremental.py ---------------------------

def list_incremental_stems_needing_embedding():
    """
    Mirrors embed_incremental.py's find_unembedded_csvs(), but listing S3
    keys instead of a local folder. Catches this run's new file (if any)
    plus any backlog left over from a previous run whose embedding step
    failed partway -- same "safe to rerun as often as you like" guarantee
    as the original script.
    """
    incremental = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=INCREMENTAL_PREFIX)
    csv_stems = {
        obj["Key"][len(INCREMENTAL_PREFIX):-4]
        for obj in incremental.get("Contents", [])
        if obj["Key"].endswith(".csv")
    }

    embedded = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=EMBEDDINGS_INCREMENTAL_PREFIX)
    embedded_stems = {
        obj["Key"][len(EMBEDDINGS_INCREMENTAL_PREFIX):-4]
        for obj in embedded.get("Contents", [])
        if obj["Key"].endswith(".npy")
    }

    return sorted(csv_stems - embedded_stems)


def embed_stem(model, stem):
    csv_key = f"{INCREMENTAL_PREFIX}{stem}.csv"
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=csv_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    texts = df["text_original"].fillna("").tolist()

    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True
    )

    npy_buffer = io.BytesIO()
    np.save(npy_buffer, embeddings)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{EMBEDDINGS_INCREMENTAL_PREFIX}{stem}.npy",
        Body=npy_buffer.getvalue(),
    )

    meta_buffer = io.StringIO()
    df[["link", "date", "title_original", "category"]].to_csv(meta_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{EMBEDDINGS_INCREMENTAL_PREFIX}{stem}_meta.csv",
        Body=meta_buffer.getvalue().encode("utf-8"),
    )

    return embeddings.shape


# --- Entry point ------------------------------------------------------------

def lambda_handler(event, context):
    seen_links = load_seen_links()
    print(f"Loaded {len(seen_links):,} known links from s3://{BUCKET_NAME}/{SEEN_LINKS_KEY}")

    new_rows, n_fetched, skipped_no_content, skipped_already_seen = fetch_new_articles(seen_links)

    if new_rows:
        key = save_incremental_csv(new_rows)
        print(f"Saved {len(new_rows)} new article(s) to s3://{BUCKET_NAME}/{key}")

        seen_links |= {row["link"] for row in new_rows}
        save_seen_links(seen_links)
        print(f"Updated s3://{BUCKET_NAME}/{SEEN_LINKS_KEY} ({len(seen_links):,} links)")
    else:
        print("No new articles this run.")

    print(
        f"Ingestion summary: {n_fetched} fetched, {len(new_rows)} new, "
        f"{skipped_no_content} skipped (no content yet), "
        f"{skipped_already_seen} already seen"
    )

    stems_to_embed = list_incremental_stems_needing_embedding()
    if not stems_to_embed:
        print("No incremental files need embedding.")
        return {"new_articles": len(new_rows), "embedded_files": 0}

    print(f"Loading model {MODEL_NAME} (from the image, no download) ...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH

    for stem in stems_to_embed:
        start = time.time()
        shape = embed_stem(model, stem)
        elapsed = time.time() - start
        print(f"Embedded {stem}: shape {shape} in {elapsed:.1f}s")

    return {"new_articles": len(new_rows), "embedded_files": len(stems_to_embed)}
