import os
import pickle
import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

_openai_client = None
_reranker = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client

def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_bm25_index(kb_path: str, save_path: str) -> BM25Okapi:
    """Build a BM25 index from the text_chunk column and save it to disk."""
    df = pd.read_csv(kb_path)
    corpus = [str(doc).lower().split() for doc in df["text_chunk"]]
    bm25 = BM25Okapi(corpus)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {save_path} ({len(corpus)} documents)")
    return bm25


def load_bm25_index(path: str) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# System loader
# ---------------------------------------------------------------------------

def load_retrieval_system(index_path: str, kb_path: str, bm25_path: str | None = None):
    """Load the FAISS index, knowledge base DataFrame, and optionally the BM25 index.

    Returns (faiss_index, df, bm25_index).  bm25_index is None when not requested.
    """
    faiss_index = faiss.read_index(index_path)
    df = pd.read_csv(kb_path)

    bm25_index = None
    if bm25_path is not None:
        if not os.path.exists(bm25_path):
            print(f"BM25 index not found at {bm25_path}, building it now …")
            bm25_index = build_bm25_index(kb_path, bm25_path)
        else:
            bm25_index = load_bm25_index(bm25_path)

    return faiss_index, df, bm25_index


# ---------------------------------------------------------------------------
# Individual retrieval methods
# ---------------------------------------------------------------------------

def retrieve_dense(query: str, index, df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Retrieve top-k songs using FAISS dense (cosine) search."""
    response = _get_openai_client().embeddings.create(
        input=[query], model="text-embedding-3-small"
    )
    query_vec = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)
    results = df.iloc[indices[0]].copy()
    results["retrieval_score"] = scores[0]
    return results.reset_index(drop=True)


def retrieve_bm25(query: str, bm25_index: BM25Okapi, df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Retrieve top-k songs using BM25 keyword search."""
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    results = df.iloc[top_indices].copy()
    results["retrieval_score"] = scores[top_indices]
    return results.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def rerank(query: str, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
    """Rerank candidate songs with a cross-encoder and return the top-k."""
    model = _get_reranker()
    pairs = [(query, str(row["text_chunk"])) for _, row in candidates.iterrows()]
    cross_scores = model.predict(pairs)
    candidates = candidates.copy()
    candidates["rerank_score"] = cross_scores
    candidates = candidates.sort_values("rerank_score", ascending=False)
    return candidates.head(k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unified retrieve()
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    index,
    df: pd.DataFrame,
    k: int = 20,
    method: str = "dense",
    bm25_index: BM25Okapi | None = None,
) -> pd.DataFrame:
    """Unified retrieval interface.

    method options
    --------------
    "dense"        — FAISS cosine similarity only
    "bm25"         — BM25 keyword search only
    "dense+rerank" — FAISS (k*3 candidates) → cross-encoder rerank → top-k
    "bm25+rerank"  — BM25  (k*3 candidates) → cross-encoder rerank → top-k
    """
    if method == "dense":
        return retrieve_dense(query, index, df, k=k)

    elif method == "bm25":
        if bm25_index is None:
            raise ValueError("bm25_index is required for method='bm25'")
        return retrieve_bm25(query, bm25_index, df, k=k)

    elif method == "dense+rerank":
        candidates = retrieve_dense(query, index, df, k=k * 3)
        return rerank(query, candidates, k=k)

    elif method == "bm25+rerank":
        if bm25_index is None:
            raise ValueError("bm25_index is required for method='bm25+rerank'")
        candidates = retrieve_bm25(query, bm25_index, df, k=k * 3)
        return rerank(query, candidates, k=k)

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: 'dense', 'bm25', 'dense+rerank', 'bm25+rerank'."
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    INDEX_PATH = "data/index/songs.index"
    KB_PATH    = "data/index/knowledge_base.csv"
    BM25_PATH  = "data/index/bm25.pkl"

    # Build BM25 index if it doesn't exist yet
    if not os.path.exists(BM25_PATH):
        build_bm25_index(KB_PATH, BM25_PATH)

    faiss_index, df, bm25_idx = load_retrieval_system(INDEX_PATH, KB_PATH, BM25_PATH)

    QUERY = "upbeat 1980s pop songs for a birthday celebration"
    K = 5

    for method in ("dense", "bm25", "dense+rerank", "bm25+rerank"):
        print(f"\n{'='*60}")
        print(f"Method: {method}  |  query: \"{QUERY}\"  |  k={K}")
        print('='*60)
        results = retrieve(QUERY, faiss_index, df, k=K, method=method, bm25_index=bm25_idx)
        for i, row in results.iterrows():
            score_col = "rerank_score" if "rerank" in method else "retrieval_score"
            score = row.get(score_col, "—")
            print(f"  {i+1}. {row['song']} — {row['artist']} ({row['year']})  [{score:.4f}]")
