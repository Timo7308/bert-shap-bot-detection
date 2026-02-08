------------------------------------------------------------------------------------
# Purpose of this script:
# - Convert large raw tweet JSON files into a compact and efficient Parquet format.
# - Process the data in chunks to handle very large files without exhausting memory.
#
# Important note:
# - The conversion preserves only the fields required for later analysis.
# - Chunking and merging are technical steps to enable scalable data processing,
#   not analytical decisions.
-------------------------------------------------------------------------------------

import ijson
import pandas as pd
import os
import glob

def convert_json_to_parquet(json_path, final_parquet_path, chunk_size=500_000):
    
    temp_dir = "temp_parquet_chunks"
    os.makedirs(temp_dir, exist_ok=True)

    rows = []
    total_count = 0
    file_count = 0

    # Convert JSON into parquet chunks
    with open(json_path, "r", encoding="utf-8") as f:
        parser = ijson.items(f, "item")
        for tweet in parser:
            try:
                row = {
                    "author_id": str(tweet.get("author_id") or tweet.get("user_id") or (tweet.get("user") or {}).get("id")).lstrip("u"),
                    "text": tweet.get("text", ""),
                    "created_at": tweet.get("created_at", "")
                }
                rows.append(row)
                total_count += 1
            except Exception:
                continue

            if len(rows) >= chunk_size:
                file_count += 1
                chunk_file = os.path.join(temp_dir, f"chunk_{file_count}.parquet")
                pd.DataFrame(rows).to_parquet(chunk_file, index=False, engine="pyarrow", compression="snappy")
                rows.clear()
                print(f"Chunk {file_count} gespeichert ({total_count:,} Zeilen bisher)")

        
        if rows:
            file_count += 1
            chunk_file = os.path.join(temp_dir, f"chunk_{file_count}.parquet")
            pd.DataFrame(rows).to_parquet(chunk_file, index=False, engine="pyarrow", compression="snappy")

    print(f"Chunks erstellt: {file_count} Dateien mit insgesamt {total_count:,} Zeilen.")

    # Merge all chunks into a single file
    print("Mergen aller Chunks in eine Datei...")
    all_chunks = []
    for chunk_path in glob.glob(os.path.join(temp_dir, "*.parquet")):
        all_chunks.append(pd.read_parquet(chunk_path))

    final_df = pd.concat(all_chunks, ignore_index=True)
    final_df.to_parquet(final_parquet_path, index=False, engine="pyarrow", compression="snappy")

    print(f"Fertig! {len(final_df):,} Zeilen gespeichert in {final_parquet_path}")

    # Delete temp folder
    for file in glob.glob(os.path.join(temp_dir, "*.parquet")):
        os.remove(file)
    os.rmdir(temp_dir)

# Manually set input and output file beacause of the large size of each file
if __name__ == "__main__":
    input_file = "tweet.json"
    output_file = "tweet.parquet"
    convert_json_to_parquet(input_file, output_file)
