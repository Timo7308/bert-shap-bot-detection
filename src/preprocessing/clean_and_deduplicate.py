import pandas as pd
import re
import csv

INPUT_CSV  = "twibot1.csv"
OUTPUT_CSV = "twibot_pre.csv"

MIN_WORDS = 25
REMOVE_EMOJIS = True

TEXT_COL = "text"
LABEL_COL = "label"
AUTHOR_COL = "author_id"

def read_csv_robust(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        engine="python",
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip"
    )

df = read_csv_robust(INPUT_CSV)
print(f"Loaded rows: {len(df):,}")

# Cheap pre-filter
df[TEXT_COL] = df[TEXT_COL].astype(str)
df = df[df[TEXT_COL].str.strip().astype(bool)].copy()
raw_wc = df[TEXT_COL].str.split().str.len()
df = df[raw_wc >= MIN_WORDS].copy()
print(f"After raw >= {MIN_WORDS} words: {len(df):,}")

# Compile regex once
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000027BF"
    "]+",
    flags=re.UNICODE
)
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
USER_RE = re.compile(r"@\w+")
RT_RE   = re.compile(r"^\s*rt\s+@\w+:?\s*", re.IGNORECASE)
KEEP_RE = re.compile(r"[^\w\s'.,!?\-]", re.UNICODE)

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.strip()
    t = RT_RE.sub("", t)
    t = t.lower()
    t = URL_RE.sub(" HTTPURL ", t)
    t = USER_RE.sub(" @USER ", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    if REMOVE_EMOJIS:
        t = EMOJI_RE.sub(" ", t)
    t = KEEP_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

texts = df[TEXT_COL].tolist()
df["clean_text"] = [clean_text(x) for x in texts]
df = df[df["clean_text"].astype(bool)].copy()

df["word_count"] = df["clean_text"].str.split().str.len()
before = len(df)
df = df[df["word_count"] >= MIN_WORDS].copy()
print(f"After clean >= {MIN_WORDS} words: {len(df):,} (removed {before - len(df):,})")

# Dedup early 
before_dup = len(df)
df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
print(f"After dedup: {len(df):,} (removed {before_dup - len(df):,})")

df_out = df[[LABEL_COL, "clean_text", AUTHOR_COL]]
df_out.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_ALL,
    escapechar="\\"
)
print(f"Saved prefiltered dataset to {OUTPUT_CSV} with {len(df_out):,} rows.")



