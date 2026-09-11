"""
Download the seed corpus for the Ukraine news search project.

Source: "Multilingual news dataset about Ukraine (2022-2025)"
Authors: Hennadii Bohuta, Ihor Ihnatiev, Khrystyna Lipianina-Honcharenko
Published: Figshare, DOI 10.6084/m9.figshare.29020670 (v3, posted 2025-06-28)
Paper: https://www.nature.com/articles/s41597-026-07033-5 (Scientific Data)
Licence: CC BY 4.0 (attribution required; commercial use and derivatives ARE
permitted under this licence -- see README for why this differs from an
earlier assumption of CC BY-NC-ND 4.0).

File: news_2022-2025.csv, ~1.02 GB, ~120,617 rows.
Columns: link, date, title_original, text_original, title_ukrainian,
         text_ukrainian, category

Usage:
    python scripts/fetch_data.py [output_path]

Note: run this from a normal internet connection / your own machine.
Figshare is not reachable from some sandboxed / restricted-egress
environments -- if the request hangs or is refused, download the file
directly from the dataset page instead:
    https://doi.org/10.6084/m9.figshare.29020670.v3
(click "news_2022-2025.csv" / the Download button) and save it to
data/raw/news_2022-2025.csv.
"""

import sys
import pathlib

import requests

FIGSHARE_DIRECT_URL = "https://figshare.com/ndownloader/files/55740404"
DEFAULT_OUTPUT = "data/raw/news_2022-2025.csv"

# Figshare's endpoint has been observed to return an empty 200 response to
# requests with no User-Agent (the default for urllib/requests). A normal
# browser-style header avoids that.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )
}


def download(url: str, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  -> {out_path}")

    with requests.get(url, headers=HEADERS, stream=True, allow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        if total == 0:
            print("  WARNING: server did not report a content-length -- "
                  "can't show a percentage, but will still stream the body.")

        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(100, downloaded * 100 // total)
                    sys.stdout.write(f"\r  {pct}%  ({downloaded/1e6:.0f} MB / {total/1e6:.0f} MB)")
                else:
                    sys.stdout.write(f"\r  {downloaded/1e6:.0f} MB downloaded")
                sys.stdout.flush()

    final_size = out_path.stat().st_size
    print(f"\nDone. Wrote {final_size/1e6:.1f} MB to {out_path}")

    if final_size < 100_000_000:  # sanity check: expect ~1.02 GB
        print(
            f"WARNING: expected ~1.02 GB (1,020,000,000 bytes) but only got "
            f"{final_size:,} bytes. This is almost certainly not the full "
            f"dataset -- open the file and check whether it's an HTML/error "
            f"page rather than CSV rows, or download it manually from "
            f"https://doi.org/10.6084/m9.figshare.29020670.v3 instead."
        )


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(DEFAULT_OUTPUT)
    try:
        download(FIGSHARE_DIRECT_URL, out)
    except requests.exceptions.RequestException as e:
        print(f"\nDownload failed: {e}")
        print(
            "Try downloading manually instead: "
            "https://doi.org/10.6084/m9.figshare.29020670.v3 "
            "(click the file name / Download button) and save it to "
            f"{out}"
        )
        sys.exit(1)
