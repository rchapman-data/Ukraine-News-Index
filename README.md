# Ukraine News Search & RAG Pipeline

Sparta Global end-of-course interview show-and-tell project. Ingests a large
real-world news corpus about the Ukraine war, proves out approximate nearest-
neighbour search (FAISS IVF/HNSW) at a scale where it actually matters, keeps
the corpus growing via a scheduled cloud job, and layers a grounded RAG
question-answering interface on top.

Status: **Part 1 in progress** - sourcing the corpus and standing up AWS
foundations (IAM user, budget alert, S3 bucket).

## Data source & attribution

**Dataset:** "Multilingual news dataset about Ukraine (2022-2025)"
**Authors:** Hennadii Bohuta, Ihor Ihnatiev, Khrystyna Lipianina-Honcharenko
**Repository:** Figshare, DOI [10.6084/m9.figshare.29020670](https://doi.org/10.6084/m9.figshare.29020670) (v3, posted 2025-06-28)
**Paper:** Bohuta et al., *Scientific Data* (2026), [https://www.nature.com/articles/s41597-026-07033-5](https://www.nature.com/articles/s41597-026-07033-5)
**Licence: CC BY 4.0** (attribution required; commercial use and derivative
works are permitted).

> Note: the Figshare item page for this dataset currently lists the licence
> as **CC BY 4.0**, not CC BY-NC-ND 4.0. Attribution is still required either
> way, so nothing here changes on that front - but CC BY 4.0 is materially
> more permissive (it allows commercial use and derivative/modified
> redistribution), so the "don't publicly redistribute a modified version"
> constraint doesn't actually apply. Worth double-checking the licence field
> yourself on the dataset page before relying on this, since it's the kind
> of detail that's easy to get right generally but is important to verify.
> Either way, not committing the raw CSV to the repo remains good practice
> (it's a 1GB+ file, not project code) - `scripts/fetch_data.py` re-downloads
> it instead.

120,617 news articles, Feb 2022 - 2025, columns: `link`, `date`,
`title_original`, `text_original`, `title_ukrainian`, `text_ukrainian`,
`category`. Over 80% sourced from Ukrainska Pravda and sister sites.

**Live extension:** [Ukrainska Pravda's RSS feeds](https://www.pravda.com.ua/eng/rss-info/)
provide full article bodies in `<content:encoded>`, letting a scheduled job
keep extending this exact corpus (see later build steps). Ukrainska Pravda
requires attribution with a hyperlink no lower than the third paragraph for
reused material.

## Getting the data

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # downloads to data/raw/news_2022-2025.csv
python scripts/validate_data.py       # sanity-checks row count, columns, dupes
```

If the download script can't reach Figshare from your environment, download
`news_2022-2025.csv` directly from the [dataset page](https://doi.org/10.6084/m9.figshare.29020670.v3)
in a browser and place it at `data/raw/news_2022-2025.csv`.

## AWS foundations

See [`docs/aws-setup.md`](docs/aws-setup.md) for the IAM user, budget alert,
and S3 bucket setup steps.

## Project layout

```
scripts/          one-off / setup scripts (fetch, validate)
docs/             setup guides, architecture notes, Q&A prep
data/raw/         gitignored - the untouched downloaded CSV
data/processed/   gitignored - cleaned data, SQL exports
```

## Roadmap

1. Source corpus + AWS foundations *(this part)*
2. Validate and clean the data
3. Load structured metadata into SQL + write real queries
4. Embed the corpus, persist embeddings to S3
5. Brute-force search + FAISS ANN index (IVF/HNSW), benchmark at scale
6. Scheduled cloud ingestion (Lambda + EventBridge, parsing RSS)
7. Light RAG layer over the combined corpus
8. Polish: README, architecture diagram, benchmark chart, demo clip
9. Talking points + Q&A prep doc for the interview
