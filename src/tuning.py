import json
import time
import pandas as pd
from src.retrieval import load_retrieval_system, retrieve
from src.profiling import generate_queries, profile_to_context
from src.generation import generate_playlist
from src.evaluation import (
    precision_at_k, recall, mrr,
    historical_plausibility, llm_judge
)
from configs.prompts import GENERATION_PROMPT

# Hyperparameter grid
RETRIEVAL_METHODS = ["dense", "bm25", "dense+rerank", "bm25+rerank"]
K_VALUES = [10, 20, 30, 50]


def tune_variant_b(val_profiles, faiss_index, bm25, df):
    """Grid search over method and k for Variant B on validation profiles."""
    results = []

    for method in RETRIEVAL_METHODS:
        for k in K_VALUES:
            print(f"\n--- Variant B | method={method} | k={k} ---")
            scores = []

            for profile in val_profiles:
                bump_start = profile["birth_year"] + 15
                bump_end = profile["birth_year"] + 25
                query = f"popular songs {bump_start}-{bump_end}"

                try:
                    retrieved = retrieve(query, faiss_index, df, k=k, method=method, bm25_index=bm25)
                    result = generate_playlist(profile, retrieved, GENERATION_PROMPT)
                    hist_score = historical_plausibility(
                        result["playlist"], df, bump_start, bump_end
                    )
                    judge = llm_judge(profile, result["playlist"])
                    scores.append({
                        "hist_plausibility": hist_score,
                        "bio_precision_llm": judge["biographical_precision"],
                        "cultural_approp_llm": judge["cultural_appropriateness"],
                        "overall_llm": judge["overall_quality"],
                    })
                except Exception as e:
                    print(f"  Skipping {profile.get('id', '?')} due to error: {e}")

            avg = pd.DataFrame(scores).mean().to_dict()
            results.append({
                "variant": "B",
                "method": method,
                "k": k,
                **{f"avg_{key}": val for key, val in avg.items()}
            })
            print(f"  Avg scores: {avg}")

    return pd.DataFrame(results)


def tune_variant_c(val_profiles, faiss_index, bm25, df):
    """Grid search over method and k for Variant C on validation profiles."""
    results = []

    for method in RETRIEVAL_METHODS:
        for k in K_VALUES:
            print(f"\n--- Variant C | method={method} | total_k={k} ---")
            scores = []

            for profile in val_profiles:
                bump_start = profile["birth_year"] + 15
                bump_end = profile["birth_year"] + 25

                try:
                    retrieved, queries = profile_to_context(
                        profile, faiss_index, bm25, df,
                        method=method, k_per_query=max(k // 5, 5), total_k=k
                    )
                    result = generate_playlist(profile, retrieved, GENERATION_PROMPT)
                    hist_score = historical_plausibility(
                        result["playlist"], df, bump_start, bump_end
                    )
                    judge = llm_judge(profile, result["playlist"])
                    scores.append({
                        "hist_plausibility": hist_score,
                        "bio_precision_llm": judge["biographical_precision"],
                        "cultural_approp_llm": judge["cultural_appropriateness"],
                        "overall_llm": judge["overall_quality"],
                    })
                except Exception as e:
                    print(f"  Skipping {profile.get('id', '?')} due to error: {e}")

            avg = pd.DataFrame(scores).mean().to_dict()
            results.append({
                "variant": "C",
                "method": method,
                "k": k,
                **{f"avg_{key}": val for key, val in avg.items()}
            })
            print(f"  Avg scores: {avg}")

    return pd.DataFrame(results)
