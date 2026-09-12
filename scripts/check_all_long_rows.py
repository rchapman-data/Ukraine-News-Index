import pandas as pd
from collections import Counter

df = pd.read_csv("data/processed/flagged_long.csv")
df["text_len"] = df["text_original"].str.len()

REPEAT_THRESHOLD = 10

def chunk_stats(text):
    chunks = [c.strip() for c in text.split(".") if len(c.strip()) > 20]
    if not chunks:
        return pd.Series({"dup_ratio": 1.0, "max_repeat": 0, "highly_repeated_chunks": 0})
    counts = Counter(chunks)
    highly_repeated = sum(1 for c, n in counts.items() if n >= REPEAT_THRESHOLD)
    return pd.Series({
        "dup_ratio": len(set(chunks)) / len(chunks),
        "max_repeat": counts.most_common(1)[0][1],
        "highly_repeated_chunks": highly_repeated,
    })

stats = df["text_original"].astype(str).apply(chunk_stats)
df = pd.concat([df, stats], axis=1)

print(df.sort_values("highly_repeated_chunks", ascending=False)
      .head(25)[["link", "date", "text_len", "dup_ratio", "max_repeat", "highly_repeated_chunks"]])

for threshold in [0, 1, 2, 5, 10, 20]:
    count = (df["highly_repeated_chunks"] > threshold).sum()
    print(f"Rows with more than {threshold} highly-repeated chunks: {count}")