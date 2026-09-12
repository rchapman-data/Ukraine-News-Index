import pandas as pd

# Load the flagged long rows
df = pd.read_csv("data/processed/flagged_long.csv")

# Recreate the length column (in case it wasn't saved into the CSV itself)
df["text_len"] = df["text_original"].str.len()

# Show the 10 longest, so you can confirm which one you're investigating
print(df.sort_values("text_len", ascending=False).head(10)[["link", "date", "text_len"]])
print()

# Grab the single longest row
# row = df.sort_values("text_len", ascending=False).iloc[0]
row = df[df["link"].str.contains("7160484")].iloc[0]  # use whatever ID is in that row's link
text = row["text_original"]

print(f"Investigating: {row['link']}")
print(f"Length: {len(text)} characters")
print()

# Save the full text to a plain file so you can read all of it, no character limit
with open("row_check.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Full text written to row_check.txt")
print()

# Check for repetition: split into sentence-like chunks and count duplicates
chunks = [c.strip() for c in text.split(".") if len(c.strip()) > 20]
print(f"Total sentence-like chunks: {len(chunks)}")
print(f"Unique chunks: {len(set(chunks))}")

from collections import Counter
counts = Counter(chunks)
print()
print("Most repeated chunks:")
for chunk, n in counts.most_common(5):
    print(f"{n}x -- {chunk[:100]}")