"""
Local dry run for Part 6's RSS ingestion: fetch Ukrainska Pravda's
Ukrainian-language "News" feed, extract each article's actual text, and
report what comes back -- before any AWS/Lambda wiring.

Feed chosen: https://www.pravda.com.ua/rss/view_news/ ("Новини" / News),
not the Publications feed or the /eng/ edition -- see build-progress.md for
the reasoning (News feed reliably has full content:encoded when content is
present at all; the Ukrainian-language feed keeps new articles in the same
primary language as the bulk of the existing 120,582-article corpus).

Two real quirks discovered by hand-inspecting the raw feed, both handled
below:

1. `content:encoded` is HTML (paragraphs, spans, links, sometimes a
   `<template>` tag holding a hidden glossary popover definition that must
   NOT be read as real body text) -- needs converting to plain text to match
   the corpus's existing `text_original` column.

2. Some items -- seemingly ones without a named `dc:creator` -- have
   completely empty `content:encoded` and `description`. Rather than guess
   why, this script treats "too short after cleaning" as "not available
   yet" and skips the item. Because dedup happens on `link` and only after
   an item is actually kept, a skipped item just gets checked again on the
   next scheduled run -- it isn't marked as done.

Also strips a known boilerplate pattern (a trailing "support us on Patreon"
blockquote seen in the /eng/ feed during investigation) if present, the
same principled way Part 2 handled the donor-disclaimer boilerplate found
in the seed corpus.

Run this locally and read the printed output -- nothing is saved to disk or
S3 yet. This is purely "does the parsing work on real data."
"""

import feedparser
from bs4 import BeautifulSoup

RSS_FEED_URL = "https://www.pravda.com.ua/rss/view_news/"

# Below this many characters, treat the article as "not available yet" and
# skip it rather than ingesting a near-empty row.
MIN_CONTENT_CHARS = 50

# If any of these show up in the cleaned text, treat it and everything after
# it as trailing boilerplate and cut it off.
BOILERPLATE_MARKERS = ["Patreon", "patreon.com"]


def clean_html_to_text(html_fragment):
    """
    Convert an HTML fragment (from content:encoded) into plain text,
    matching the style of the corpus's existing text_original column.
    """
    soup = BeautifulSoup(html_fragment, "html.parser")

    # <template> holds hidden glossary-popover text (only shown via JS on
    # hover/click) -- not real body text, must not be spliced into it.
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


def main():
    print(f"Fetching {RSS_FEED_URL} ...")
    feed = feedparser.parse(RSS_FEED_URL)
    print(f"bozo (parse had trouble?): {feed.bozo}")
    print(f"Found {len(feed.entries)} items\n")

    kept = 0
    skipped = 0

    for entry in feed.entries:
        raw_html = ""
        if "content" in entry and len(entry.content) > 0:
            raw_html = entry.content[0].value

        text = clean_html_to_text(raw_html)
        text = strip_trailing_boilerplate(text)

        if len(text) < MIN_CONTENT_CHARS:
            skipped += 1
            print(f"[SKIP - no content yet] {entry.title}")
            continue

        kept += 1
        print(f"[OK] {entry.title}")
        print(f"     link: {entry.link}")
        print(f"     category: {entry.get('category', '(none)')}")
        print(f"     published: {entry.get('published', '(none)')}")
        print(f"     text ({len(text)} chars): {text[:150]}...")
        print()

    print(f"\nSummary: {kept} kept, {skipped} skipped (empty/too-short content)")


if __name__ == "__main__":
    main()
