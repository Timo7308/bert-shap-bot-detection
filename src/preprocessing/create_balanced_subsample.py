--------------------------------------------------------------------------------------
# Purpose of this script:
# - Create a large, balanced tweet sample from the original Twibot-22 dataset.
# - Map author-level labels (human/bot) to tweet-level data.
# - Randomly sample an equal number of bot and human tweets.
#
# Important note:
# - Labels are assigned at the author level and inherited by all tweets of that author.
# - The resulting dataset is a balanced subsample used for analysis and model training,
#   not a representation of real-world class distributions.
---------------------------------------------------------------------------------------


#Creating main random sample from all the data
import pandas as pd
import random

SEED = 42
random.seed(SEED)

print("Step 1: Load labels...")
label_df = pd.read_csv("label.csv")
label_df["id"] = label_df["id"].astype(str).str.lstrip("u")
LABEL_MAP = dict(zip(label_df["id"], label_df["label"]))

print("Step 2: Load parquet data...")
files_to_process = [
    "tweet_0.parquet", "tweet_1.parquet", "tweet_2.parquet",
    "tweet_3.parquet", "tweet_4.parquet", "tweet_5.parquet",
    "tweet_6.parquet", "tweet_7.parquet", "tweet_8.parquet"
]

all_data = []
for file_path in files_to_process:
    df = pd.read_parquet(file_path)
    df["author_id"] = df["author_id"].astype(str).str.lstrip("u")
    df["label"] = df["author_id"].map(LABEL_MAP)
    df = df[df["label"].isin(["human", "bot"])]
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
print(f"Gesamt: {len(df_all):,} Einträge")


bots = df_all[df_all["label"] == "bot"]
humans = df_all[df_all["label"] == "human"]

print(f"Tweets Bots: {len(bots):,}")
print(f"Tweets Humans: {len(humans):,}")

# Targetsize 500,000 tweets per class
target_size = 500_000
print(f"Sampling {target_size} Tweets per class...")

# Sample at random from both classes
bots_sample = bots.sample(n=target_size, random_state=SEED)
humans_sample = humans.sample(n=target_size, random_state=SEED)

# Bring both classes together and shuffle
final_df = pd.concat([bots_sample, humans_sample])
final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

final_df.to_csv("twibot1.csv", index=False)
print(f"Done! {len(final_df)} Saved in twibot1.csv")
