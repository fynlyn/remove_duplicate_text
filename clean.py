import math
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.feature_extraction import FeatureHasher
from sklearn.neighbors import NearestNeighbors

# ===== CONFIG =====
INPUT_FILE = "huge_file.xlsx"
SHEET_NAME = 0
TEXT_COLUMN = "text"
N_FEATURES = 15
SIM_THRESHOLD = 0.1    # normalized L1 threshold (0..1). raw L1 radius = SIM_THRESHOLD * N_FEATURES
CHUNKS = 20            # how many chunks to split the DF for parallel hashing
QUERY_BATCH = 5000     # how many query points per radius_neighbors call (tuneable)
# ===================

def text_to_dict(text):
    tokens = str(text).split()
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return freq

def process_chunk(df_chunk):
    """Worker: convert text column -> frequency dicts -> hashed features (numpy array)."""
    docs = [text_to_dict(txt) for txt in df_chunk[TEXT_COLUMN].astype(str)]
    hasher = FeatureHasher(n_features=N_FEATURES, input_type="dict")
    X = hasher.transform(docs)        # sparse matrix
    return X.toarray()                # dense small matrix (chunk_size x N_FEATURES)

def parallel_hashing(df):
    """
    Split df into CHUNKS, inside each chunk split across CPUs.
    Append results chunk-by-chunk to avoid exploding memory in pool.map.
    Returns full X (num_rows x N_FEATURES) as float32.
    """
    n = len(df)
    chunk_size = math.ceil(n / CHUNKS)
    X_parts = []

    for start in range(0, n, chunk_size):
        df_chunk = df.iloc[start: start + chunk_size]

        # further split across CPUs inside this chunk
        sub_size = math.ceil(len(df_chunk) / cpu_count())
        subchunks = [df_chunk.iloc[i:i + sub_size] for i in range(0, len(df_chunk), sub_size)]

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_chunk, subchunks)

        X_chunk = np.vstack(results)
        X_parts.append(X_chunk)

    X = np.vstack(X_parts)
    # convert to float32 to save memory (sklearn supports float32)
    return X.astype(np.float32)

def deduplicate(df, X, threshold=SIM_THRESHOLD, query_batch=QUERY_BATCH):
    """
    Memory-safe deduplication using radius-based nearest neighbor queries.

    - Build a NearestNeighbors index with metric='manhattan'.
    - For each batch of query points, call radius_neighbors to find neighbors within radius.
    - Keep the first occurrence (lowest index) and mark later neighbors as removed.
    Returns kept_idx, removed_idx (numpy arrays of indices).
    """
    n = X.shape[0]
    keep_mask = np.ones(n, dtype=bool)

    # radius in raw L1 units
    radius = float(threshold * X.shape[1]) + 1e-12

    print(f"Fitting NearestNeighbors on {n} vectors (metric=manhattan)...")
    nn = NearestNeighbors(metric="manhattan", n_jobs=cpu_count())
    nn.fit(X)   # builds index (tree/balltree/hybrid depending on sklearn)

    print("Querying neighbors in batches and marking duplicates...")
    for start in range(0, n, query_batch):
        end = min(start + query_batch, n)
        batch = X[start:end]

        # return list of neighbor indices for each query point
        neighbors_list = nn.radius_neighbors(batch, radius=radius, return_distance=False)

        for i_local, neighbors in enumerate(neighbors_list):
            i = start + i_local
            if not keep_mask[i]:
                continue  # already flagged duplicate; skip

            # Only mark neighbors that come *after* current index to preserve first occurrence
            # (avoids race where two points mark each other)
            for nb in neighbors:
                if nb <= i:
                    continue
                if keep_mask[nb]:
                    keep_mask[nb] = False

        # progress simple print (optional)
        if (start // query_batch) % 10 == 0:
            print(f"Processed queries: {start} / {n}")

    kept_idx = np.where(keep_mask)[0]
    removed_idx = np.where(~keep_mask)[0]
    return kept_idx, removed_idx

if __name__ == "__main__":
    print("Reading Excel...")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    print("Hashing texts in parallel (chunked)...")
    X = parallel_hashing(df)  # returns float32 array shape (n_rows, N_FEATURES)
    print("Hashed matrix shape:", X.shape, " dtype:", X.dtype)

    print("Deduplicating rows...")
    kept_idx, removed_idx = deduplicate(df, X)

    df_clean = df.iloc[kept_idx]
    df_removed = df.iloc[removed_idx]

    print(f"Original rows: {len(df)} | Kept: {len(df_clean)} | Removed: {len(df_removed)}")

    df_clean.to_csv("cleaned_file.csv", index=False)
    df_removed.to_csv("removed_rows.csv", index=False)

    print("Saved cleaned rows to cleaned_file.csv")
    print("Saved removed rows to removed_rows.csv")
