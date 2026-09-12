import pandas as pd

df = pd.read_csv("data/processed/flagged_short.csv")

for i, row in df.iterrows():
    print("=" * 80)
    print("link:", row["link"])
    print("date:", row["date"])
    print("title:", row["title_original"])
    print("text_original:", row["text_original"])
    print()