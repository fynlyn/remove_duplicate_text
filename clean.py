import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import pairwise_distances
from multiprocessing import Pool, cpu_count
import math

# ========= CONFIG =========
INPUT_FILE = "huge_file.xlsx"
SHEET_NAME = 0
TEXT_COLUMN = "text"
N_FEATURES = 15
SIM_THRESHOLD = 0.1     # similarity threshold (0.0 = identical, higher = looser)
CHUNKS = 20             # number of parallel chunks
# ==========================


def text_to_dict(text):
    """Convert a string into a frequency dictionary for hashing."""
    tokens = text.split()
    freq_dict = {}
    for t in tokens:
        freq_dict[t] = freq_dict.get(t, 0) + 1
    return freq_dict


def process_chunk(df_chunk):
    """Hash text in one dataframe chunk."""
    docs = [text_to_dict(str(txt)) for txt in df_chunk[TEXT_COLUMN]]
    hasher = FeatureHasher(n_features=N_FEATURES, input_type="dict")
    X = hasher.transform(docs)
    return X.toarray()


def parallel_hashing(df):
    """
    Memory-safe parallel hashing:
    - Splits df into chunks
    - Processes each chunk in parallel
    - Writes partial results into a list and returns combined array
    """
    chunk_size = math.ceil(len(df) / CHUNKS)
    X_list = []

    for start in range(0, len(df), chunk_size):
        df_chunk = df.iloc[start:start + chunk_size]

        # Further split chunk across CPUs
        sub_size = math.ceil(len(df_chunk) / cpu_count())
        subchunks = [df_chunk.iloc[i:i + sub_size] for i in range(0, len(df_chunk), sub_size)]

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_chunk, subchunks)

        # Append chunk result (instead of keeping all in memory at once)
        X_list.append(np.vstack(results))

    return np.vstack(X_list)


def deduplicate(df, X, threshold=SIM_THRESHOLD):
    """
    Deduplicate rows based on hashed features.
    Returns indices of kept and removed rows.
    """
    keep_mask = np.ones(len(X), dtype=bool)

    batch_size = 5000
    for i in range(0, len(X), batch_size):
        end = min(i + batch_size, len(X))
        block = X[i:end]

        # compare block with the rest (still heavy for 1M rows, but blockwise)
        dists = pairwise_distances(block, X, metric="manhattan") / X.shape[1]

        for j, row_dists in enumerate(dists):
            idx = i + j
            if not keep_mask[idx]:
                continue  # already marked duplicate

            # find near-duplicates
            near = np.where((row_dists <= threshold) & (row_dists > 0))[0]
            keep_mask[near] = False  # mark duplicates

    kept_idx = np.where(keep_mask)[0]
    removed_idx = np.where(~keep_mask)[0]

    return kept_idx, removed_idx


if __name__ == "__main__":
    print("Reading Excel...")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    print("Hashing texts in parallel (chunked)...")
    X = parallel_hashing(df)

    print("Deduplicating rows...")
    kept_idx, removed_idx = deduplicate(df, X)

    df_clean = df.iloc[kept_idx]
    df_removed = df.iloc[removed_idx]

    print(f"Original rows: {len(df)} | Kept: {len(df_clean)} | Removed: {len(df_removed)}")

    df_clean.to_csv("cleaned_file.csv", index=False)
    df_removed.to_csv("removed_rows.csv", index=False)

    print("Saved cleaned rows to cleaned_file.csv")
    print("Saved removed rows to removed_rows.csv")
