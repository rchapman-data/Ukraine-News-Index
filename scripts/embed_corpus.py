"""
Embed the Ukraine News Index corpus using a multilingual sentence-transformers model.

Reads the cleaned corpus (news_cleaned.csv), embeds each article's text_original
field (the article in whatever language it was actually published in - this corpus
genuinely mixes Ukrainian, English, Hungarian, and others), and saves:

  - embeddings.npy      : one row per article,S embedding vectors, in the same row
                           order as the metadata file below
  - embeddings_meta.csv : link/date/title/category for each article, same row order
                           as embeddings.npy, so vectors can be joined back to their
                           source article later (by row position)

Model: paraphrase-multilingual-MiniLM-L12-v2
- Multilingual: places text from different languages into one shared vector space,
  so semantic search works across languages, not just within one.
- max_seq_length is set to 256 tokens below (the model's own default is 128). This
  model - like most sentence-transformer models - truncates anything longer than
  its max sequence length. Articles here can run to thousands of characters, so
  embeddings represent roughly the opening of each article, not the full text.
  Deliberate, documented simplification - not a bug.

Run on the EC2 GPU instance, inside the activated pytorch environment
(source /opt/pytorch/bin/activate), after installing sentence-transformers.
"""

import time
import boto3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_CSV = "news_cleaned.csv"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 128

S3_BUCKET = "rcchap79-ukraine-news-index"
S3_PREFIX = "embeddings"

EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "embeddings_meta.csv"


def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows")

    # Defensive: shouldn't be any empty text_original rows left after Part 2's
    # cleaning, but fillna keeps this script safe to rerun even if that changes.
    texts = df["text_original"].fillna("").tolist()

    print(f"Loading model {MODEL_NAME} onto GPU ...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.max_seq_length = MAX_SEQ_LENGTH

    print(f"Embedding {len(texts):,} articles (batch size {BATCH_SIZE}) ...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - start
    print(f"Done in {elapsed / 60:.1f} minutes. Embeddings shape: {embeddings.shape}")

    print(f"Saving embeddings to {EMBEDDINGS_FILE} ...")
    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"Saving metadata to {METADATA_FILE} ...")
    df[["link", "date", "title_original", "category"]].to_csv(METADATA_FILE, index=False)

    print(f"Uploading both files to s3://{S3_BUCKET}/{S3_PREFIX}/ ...")
    s3 = boto3.client("s3")
    s3.upload_file(EMBEDDINGS_FILE, S3_BUCKET, f"{S3_PREFIX}/{EMBEDDINGS_FILE}")
    s3.upload_file(METADATA_FILE, S3_BUCKET, f"{S3_PREFIX}/{METADATA_FILE}")

    print("Done.")


if __name__ == "__main__":
    main()