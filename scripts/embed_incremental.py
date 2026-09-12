"""
Embed any not-yet-embedded incremental article files produced by
scripts/ingest_rss.py, writing each one's vectors to its own matching file
under embeddings/incremental/ -- the base embeddings.npy from Part 4 is
never touched.

Same model as scripts/embed_corpus.py (paraphrase-multilingual-MiniLM-L12-v2,
max_seq_length=256) so new vectors land in the exact same shared vector
space as the historical 120,582 -- this is required for search to treat old
and new articles consistently, not a style choice.

Runs on CPU deliberately, not GPU: a scheduled run only has a handful of new
articles to embed (RSS produces a few dozen items per pull, most already
seen), so there's no computational need for a GPU/EC2 instance here the way
there was for the one-off 120k-article Part 4 job.

"Already embedded" is tracked simply: for each data/incremental/<name>.csv,
if embeddings/incremental/<name>.npy already exists, it's skipped. This
script is meant to be safe to rerun as often as you like.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_INCREMENTAL_DIR = Path("data/incremental")
EMBEDDINGS_INCREMENTAL_DIR = Path("embeddings/incremental")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 128


def find_unembedded_csvs():
    csv_paths = sorted(DATA_INCREMENTAL_DIR.glob("*.csv"))
    unembedded = []
    for csv_path in csv_paths:
        expected_npy = EMBEDDINGS_INCREMENTAL_DIR / f"{csv_path.stem}.npy"
        if not expected_npy.exists():
            unembedded.append(csv_path)
    return unembedded


def main():
    to_process = find_unembedded_csvs()
    if not to_process:
        print("No new incremental files to embed.")
        return

    print(f"Found {len(to_process)} incremental file(s) to embed:")
    for path in to_process:
        print(f"  - {path.name}")

    print(f"\nLoading model {MODEL_NAME} on CPU ...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH

    EMBEDDINGS_INCREMENTAL_DIR.mkdir(parents=True, exist_ok=True)

    for csv_path in to_process:
        df = pd.read_csv(csv_path)
        texts = df["text_original"].fillna("").tolist()

        print(f"\nEmbedding {len(texts)} article(s) from {csv_path.name} ...")
        start = time.time()
        embeddings = model.encode(
            texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True
        )
        elapsed = time.time() - start
        print(f"Done in {elapsed:.1f}s. Shape: {embeddings.shape}")

        out_npy = EMBEDDINGS_INCREMENTAL_DIR / f"{csv_path.stem}.npy"
        out_meta = EMBEDDINGS_INCREMENTAL_DIR / f"{csv_path.stem}_meta.csv"

        np.save(out_npy, embeddings)
        df[["link", "date", "title_original", "category"]].to_csv(out_meta, index=False)

        print(f"Saved {out_npy} and {out_meta}")


if __name__ == "__main__":
    main()
