"""
Benchmark brute-force search against FAISS approximate (IVF, HNSW) search,
at increasing corpus scale, to quantify the real speed/accuracy trade-off
that approximate nearest-neighbor search offers.

This is the centerpiece of Part 5: the group project's dataset (~267 jobs)
was too small for FAISS's approximate-search advantage to ever show up. At
120,582 articles, it should.

Methods compared, all using cosine similarity (vectors are L2-normalized
first, so inner product == cosine similarity):

  - brute_force : plain numpy matrix multiplication against every vector in
                  the subset, then sort. Exact -- this is the ground truth
                  every other method's recall is measured against.
  - faiss_ivf   : FAISS IndexIVFFlat. Clusters the corpus into `nlist`
                  groups ahead of time; a search only checks the `nprobe`
                  most relevant groups instead of scanning everything.
  - faiss_hnsw  : FAISS IndexHNSWFlat. A multi-layer graph where vectors
                  link to their nearest neighbors; search "walks" the graph
                  toward the answer instead of scanning linearly.

Scale and query design (deliberately nested, not independent draws):
one random shuffle of all 120,582 articles is generated once. Each scale
(5k/20k/50k/full) takes a PREFIX of that same shuffle, so the 5k subset is
genuinely contained inside the 20k subset, which is inside the 50k subset,
and so on -- any change in latency/recall across scales is due to scale
itself, not a different random sample each time.

The 200 query vectors are 200 random positions within the smallest (5k)
prefix. Because every larger scale's subset starts with that same prefix,
those same 200 positions point at the exact same 200 articles at every
scale -- so every method is tested against an identical query set,
regardless of corpus size.

Recall@10: of the 10 nearest neighbors brute-force found for a query, what
fraction did the approximate method also return in its own top 10?
Averaged across all 200 queries. 1.0 = perfect agreement with ground truth.

Run locally (no GPU needed -- this is CPU-bound index building and search,
not embedding).
"""

import math
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

# --- Config -------------------------------------------------------------

EMBEDDINGS_PATH = Path("embeddings/embeddings.npy")
METADATA_PATH = Path("embeddings/embeddings_meta.csv")
RESULTS_PATH = Path("results/benchmark_results.csv")

SCALES_REQUESTED = [5_000, 20_000, 50_000, None]  # None = full corpus
N_QUERIES = 200
TOP_K = 10
RANDOM_SEED = 42

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 40
HNSW_EF_SEARCH = 64

IVF_NPROBE_FRACTION = 0.10  # probe ~10% of clusters at query time


# --- Helpers --------------------------------------------------------------

def choose_nlist(n_vectors):
    """
    Pick a cluster count for IVF: roughly 4*sqrt(n) clusters (a common rule
    of thumb), but capped so there are still at least ~40 training points
    per cluster on average -- too many clusters relative to data makes
    training unstable and FAISS will warn about it.
    """
    target = int(4 * math.sqrt(n_vectors))
    max_safe = max(1, n_vectors // 40)
    return max(1, min(target, max_safe))


def recall_at_k(predicted, truth):
    """
    predicted, truth: (n_queries, k) arrays of neighbor indices.
    Returns the average, over queries, of |predicted ∩ truth| / k.
    """
    n_queries, k = truth.shape
    total = 0.0
    for row in range(n_queries):
        total += len(set(predicted[row]) & set(truth[row])) / k
    return total / n_queries


def brute_force_search(subset_vectors, query_vectors, k):
    start = time.perf_counter()
    # (n_queries, d) @ (d, n_subset) -> (n_queries, n_subset) similarity matrix
    sims = query_vectors @ subset_vectors.T
    top_k = np.argsort(-sims, axis=1)[:, :k]
    elapsed = time.perf_counter() - start
    return top_k, elapsed


def build_ivf(subset_vectors, nlist):
    d = subset_vectors.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    start = time.perf_counter()
    index.train(subset_vectors)
    index.add(subset_vectors)
    build_time = time.perf_counter() - start

    index.nprobe = max(1, int(nlist * IVF_NPROBE_FRACTION))
    return index, build_time


def build_hnsw(subset_vectors):
    d = subset_vectors.shape[1]
    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION

    start = time.perf_counter()
    index.add(subset_vectors)
    build_time = time.perf_counter() - start

    index.hnsw.efSearch = HNSW_EF_SEARCH
    return index, build_time


def timed_faiss_search(index, query_vectors, k):
    start = time.perf_counter()
    _, neighbor_ids = index.search(query_vectors, k)
    elapsed = time.perf_counter() - start
    return neighbor_ids, elapsed


# --- Main -----------------------------------------------------------------

def main():
    print(f"Loading {EMBEDDINGS_PATH} ...")
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    meta = pd.read_csv(METADATA_PATH)
    assert len(embeddings) == len(meta), (
        f"Row count mismatch: {len(embeddings)} embeddings vs {len(meta)} metadata rows"
    )
    n_total = len(embeddings)
    print(f"Loaded {n_total:,} embeddings, dim={embeddings.shape[1]}")

    # Cosine similarity == inner product once vectors are L2-normalized.
    faiss.normalize_L2(embeddings)

    scales = [s if s is not None else n_total for s in SCALES_REQUESTED]
    print(f"Scales to benchmark: {scales}")

    rng = np.random.default_rng(RANDOM_SEED)
    shuffle_order = rng.permutation(n_total)

    smallest_scale = min(scales)
    query_positions = rng.choice(smallest_scale, size=N_QUERIES, replace=False)

    results = []

    for scale in scales:
        print(f"\n=== Scale: {scale:,} articles ===")
        subset_indices = shuffle_order[:scale]
        subset_vectors = embeddings[subset_indices]
        # query_positions are valid row positions in every scale's subset
        # because every subset is a prefix of the same shuffle.
        query_vectors = subset_vectors[query_positions]

        # --- Brute-force (ground truth) ---
        bf_topk, bf_elapsed = brute_force_search(subset_vectors, query_vectors, TOP_K)
        bf_query_ms = bf_elapsed / N_QUERIES * 1000
        print(f"brute_force : {bf_query_ms:.3f} ms/query (build: n/a)")
        results.append({
            "scale": scale, "method": "brute_force",
            "build_time_s": 0.0, "avg_query_ms": bf_query_ms, "recall_at_10": 1.0,
        })

        # --- FAISS IVF ---
        nlist = choose_nlist(scale)
        ivf_index, ivf_build_time = build_ivf(subset_vectors, nlist)
        ivf_topk, ivf_elapsed = timed_faiss_search(ivf_index, query_vectors, TOP_K)
        ivf_query_ms = ivf_elapsed / N_QUERIES * 1000
        ivf_recall = recall_at_k(ivf_topk, bf_topk)
        print(
            f"faiss_ivf   : {ivf_query_ms:.3f} ms/query, recall@10={ivf_recall:.3f} "
            f"(nlist={nlist}, nprobe={ivf_index.nprobe}, build: {ivf_build_time:.2f}s)"
        )
        results.append({
            "scale": scale, "method": "faiss_ivf",
            "build_time_s": ivf_build_time, "avg_query_ms": ivf_query_ms,
            "recall_at_10": ivf_recall,
        })

        # --- FAISS HNSW ---
        hnsw_index, hnsw_build_time = build_hnsw(subset_vectors)
        hnsw_topk, hnsw_elapsed = timed_faiss_search(hnsw_index, query_vectors, TOP_K)
        hnsw_query_ms = hnsw_elapsed / N_QUERIES * 1000
        hnsw_recall = recall_at_k(hnsw_topk, bf_topk)
        print(
            f"faiss_hnsw  : {hnsw_query_ms:.3f} ms/query, recall@10={hnsw_recall:.3f} "
            f"(M={HNSW_M}, efSearch={HNSW_EF_SEARCH}, build: {hnsw_build_time:.2f}s)"
        )
        results.append({
            "scale": scale, "method": "faiss_hnsw",
            "build_time_s": hnsw_build_time, "avg_query_ms": hnsw_query_ms,
            "recall_at_10": hnsw_recall,
        })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
