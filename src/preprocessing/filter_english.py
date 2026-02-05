import pandas as pd
import csv
import langid

INPUT_CSV  = "twibot_pre.csv"
OUTPUT_CSV = "twibot2.csv"

LABEL_COL = "label"
AUTHOR_COL = "author_id"

df = pd.read_csv(INPUT_CSV, engine="python", encoding="utf-8", encoding_errors="replace")
print(f"Loaded prefiltered: {len(df):,}")

def is_english(text: str) -> bool:
    try:
        return langid.classify(text)[0] == "en"
    except:
        return False

texts = df["clean_text"].tolist()
mask = [is_english(t) for t in texts]

df = df[mask].copy().reset_index(drop=True)
print(f"After English filter: {len(df):,}")

df_out = df[[LABEL_COL, "clean_text", AUTHOR_COL]]
df_out.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_ALL,
    escapechar="\\"
)

print(f"Saved final dataset to {OUTPUT_CSV} with {len(df_out):,} rows.")

