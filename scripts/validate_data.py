"""
Sanity-check the downloaded corpus before doing anything else with it.

Usage:
    python scripts/validate_data.py [path_to_csv]
"""

import sys
import pathlib

import pandas as pd

EXPECTED_COLUMNS = {
    "link",
    "date",
    "title_original",
    "text_original",
    "title_ukrainian",
    "text_ukrainian",
    "category",
}
EXPECTED_ROWS_APPROX = 120_617


def main(path: pathlib.Path) -> None:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    size_mb = path.stat().st_size / 1e6
    print(f"File: {path}")
    print(f"Size: {size_mb:,.1f} MB")

    df = pd.read_csv(path)
    print(f"Rows: {len(df):,} (expected ~{EXPECTED_ROWS_APPROX:,})")
    print(f"Columns: {list(df.columns)}")

    missing = EXPECTED_COLUMNS - set(df.columns)
    extra = set(df.columns) - EXPECTED_COLUMNS
    if missing:
        print(f"WARNING: missing expected columns: {missing}")
    if extra:
        print(f"Note: extra columns present: {extra}")

    dupes = df.duplicated(subset=["link"]).sum() if "link" in df.columns else None
    if dupes is not None:
        print(f"Duplicate 'link' rows: {dupes}")

    empty_text = df["text_original"].isna().sum() if "text_original" in df.columns else None
    if empty_text is not None:
        print(f"Empty 'text_original' rows: {empty_text}")

    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        print(f"Date range: {parsed.min()} to {parsed.max()}")
        print(f"Unparseable dates: {parsed.isna().sum()}")

    print("\nLooks good to proceed if row count and columns match expectations.")


if __name__ == "__main__":
    p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("data/raw/news_2022-2025.csv")
    main(p)
