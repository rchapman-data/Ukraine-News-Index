"""
Validate and lightly clean the Ukraine News Index seed corpus.

Reads the raw CSV, removes exact duplicate rows and rows with no article
text, and flags (without removing) unusually short or long articles for a
human to look at. Flagged rows are written out to their own CSV files so
they're easy to review in Excel rather than reading wrapped terminal output.
"""

import pandas as pd
from pathlib import Path

# --- Config -----------------------------------------------------------

RAW_PATH = Path("data/raw/news_2022-2025.csv")
CLEANED_PATH = Path("data/processed/news_cleaned.csv")
FLAGGED_SHORT_PATH = Path("data/processed/flagged_short.csv")
FLAGGED_LONG_PATH = Path("data/processed/flagged_long.csv")

SHORT_THRESHOLD_CHARS = 100
LONG_THRESHOLD_CHARS = 20_000

TEXT_COLUMN = "text_original"
DEDUPE_COLUMN = "link"

KNOWN_BAD_LINKS = [
    "https://www.eurointegration.com.ua/news/2022/04/14/7137846/?fbclid=IwAR0BTIy8JL5z9eqDQxTy8H9aMk_pRAFzFlPpcmi3umYK1SVqgT_8WWribJc",
    "https://www.eurointegration.com.ua/news/2023/04/25/7160484/",
    "https://www.eurointegration.com.ua/news/2023/11/25/7174284/",
    "https://www.eurointegration.com.ua/news/2024/08/24/7192735/",
    "https://www.bbc.com/news/videos/cm2kdrnp99zo",
]



def main():
    print(f"Loaded ", end="")
    df = pd.read_csv(RAW_PATH)
    n_start = len(df)
    print(f"{n_start:,} rows from {RAW_PATH}")

    df = df[~df["link"].isin(KNOWN_BAD_LINKS)]
    n_after_known_bad = len(df)
    print(f"Removed {n_start - n_after_known_bad} known-bad rows")

    # Step 1: drop exact duplicate rows by link
    
    df = df.drop_duplicates(subset=[DEDUPE_COLUMN], keep="first")
    n_after_dupes = len(df)
    print(f"Dropped {n_after_known_bad - n_after_dupes} duplicate rows (by '{DEDUPE_COLUMN}')")

    # Step 2: drop rows where the article text is empty/blank
    is_blank = df[TEXT_COLUMN].isna() | (df[TEXT_COLUMN].astype(str).str.strip() == "")
    n_empty = is_blank.sum()
    df = df[~is_blank]
    print(f"Dropped {n_empty} rows with empty '{TEXT_COLUMN}'")

    # Step 3: flag (don't remove) suspiciously short or long articles
    text_len = df[TEXT_COLUMN].astype(str).str.len()

    short_mask = text_len < SHORT_THRESHOLD_CHARS
    long_mask = text_len > LONG_THRESHOLD_CHARS

    short_rows = df[short_mask]
    long_rows = df[long_mask]

    print(
        f"\nFlagged {len(short_rows)} rows with {TEXT_COLUMN} under "
        f"{SHORT_THRESHOLD_CHARS} characters (not removed -- see {FLAGGED_SHORT_PATH})"
    )
    print(
        f"Flagged {len(long_rows)} rows with {TEXT_COLUMN} over "
        f"{LONG_THRESHOLD_CHARS:,} characters (not removed -- see {FLAGGED_LONG_PATH})"
    )

    FLAGGED_SHORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    short_rows.to_csv(FLAGGED_SHORT_PATH, index=False)
    long_rows.to_csv(FLAGGED_LONG_PATH, index=False)

    # Step 4: save the cleaned file (flagged rows are still IN this file --
    # flagging is advisory only, nothing is auto-removed based on length)
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)

    n_end = len(df)
    print(f"\nSummary: {n_start:,} -> {n_end:,} rows ({n_empty} empty + {n_after_known_bad - n_after_dupes} dupes removed)")
    print(f"Saved cleaned dataset to {CLEANED_PATH}")
    print(
    f"\nNote: flagged short/long rows remain in {CLEANED_PATH.name} by design "
    f"-- they are not auto-removed. See {FLAGGED_SHORT_PATH.name} and "
    f"{FLAGGED_LONG_PATH.name} for the full flagged sets. Rows confirmed as "
    f"genuinely broken (not just unusually short/long) are excluded via "
    f"KNOWN_BAD_LINKS above."
    )   



if __name__ == "__main__":
    main()