import pandas as pd

CSV_PATH = "twibot1.csv"
LABEL_MAP = {"human": 0, "bot": 1}

# ---------- robust reader ----------
def read_robust_csv(path: str, usecols, chunksize=200_000):
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        engine="python",          
        sep=",",
        quotechar='"',
        encoding="utf-8",
        on_bad_lines="skip",    
        chunksize=chunksize
    ):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)

df = read_robust_csv(CSV_PATH, usecols=["author_id", "text", "created_at", "label"])

# ---------- normalize column names ----------
# Falls du im restlichen Code lieber clean_text nutzt:
df = df.rename(columns={"text": "clean_text"})

# ---------- basic sanity checks (RAW, keine Drops) ----------
n_rows = len(df)

# Missing in required columns (robust falls Spalten doch fehlen)
required_cols = ["clean_text", "label", "author_id"]
for c in required_cols:
    if c not in df.columns:
        df[c] = pd.NA

missing_any = df[required_cols].isna().any(axis=1).sum()
empty_text = df["clean_text"].astype(str).str.strip().eq("").sum()

# Label mapping
df["label_num"] = df["label"].map(LABEL_MAP)
unmapped_labels = df["label_num"].isna().sum()

# Für Statistik: nur Zeilen mit gültigem Label (Text/Author bleiben RAW, keine Filter außer Label-Validität)
df_l = df.dropna(subset=["label_num"]).copy()
df_l["label_num"] = df_l["label_num"].astype(int)
df_l["author_id"] = df_l["author_id"].astype(str)

# ---------- core counts ----------
n_tweets = len(df_l)
n_authors = df_l["author_id"].nunique()

tweets_per_class = (
    df_l["label_num"]
      .value_counts()
      .rename(index={0: "human", 1: "bot"})
)

authors_per_class = (
    df_l.groupby("label_num")["author_id"]
       .nunique()
       .rename(index={0: "human", 1: "bot"})
)

# ---------- tweets per author ----------
tpa = df_l.groupby("author_id").size()
tpa_mean = tpa.mean() if len(tpa) else 0
tpa_max = int(tpa.max()) if len(tpa) else 0

# ---------- text length (words) ----------
words = df_l["clean_text"].astype(str).str.split().apply(len)
words_mean = words.mean() if len(words) else 0

# ---------- output ----------
print("=== SAMPLE OVERVIEW (RAW CSV READ) ===")
print(f"Rows parsed:     {n_rows:,}")
print(f"Rows with missing fields:        {missing_any:,}")
print(f"Rows with empty text:            {empty_text:,}")
print(f"Rows with unmapped label:        {unmapped_labels:,}\n")

print("=== COUNTS (VALID LABELS ONLY) ===")
print(f"Tweets:                          {n_tweets:,}")
print(f"Unique authors:                  {n_authors:,}\n")

print("=== TWEETS PER CLASS ===")
print(tweets_per_class.to_string(), "\n")

print("=== AUTHORS PER CLASS ===")
print(authors_per_class.to_string(), "\n")

print("=== TWEETS PER AUTHOR ===")
print(f"Mean tweets/author:              {tpa_mean:.2f}")
print(f"Max tweets/author:               {tpa_max:,}\n")

print("=== TEXT LENGTH (WORDS) ===")
print(f"Mean words/tweet:                {words_mean:.2f}")
