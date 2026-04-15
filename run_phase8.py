"""
Phase 8: Hyperparameter Tuning on Validation Set
Run directly in your terminal: python3 run_phase8.py
"""
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import json, random, os
import pandas as pd
from src.retrieval import load_retrieval_system, retrieve
from src.profiling import profile_to_context
from src.generation import generate_playlist
from src.evaluation import historical_plausibility, llm_judge
from configs.prompts import GENERATION_PROMPT

RETRIEVAL_METHODS = ["dense", "bm25", "dense+rerank", "bm25+rerank"]
K_VALUES = [10, 20, 30, 50]

os.makedirs("outputs", exist_ok=True)

print("Loading retrieval system...")
faiss_index, df, bm25 = load_retrieval_system(
    "data/index/songs.index",
    "data/processed/knowledge_base.csv",
    "data/index/bm25.pkl"
)
print(f"Loaded {faiss_index.ntotal} vectors, {len(df)} songs")

with open("data/processed/val_profiles.json") as f:
    val_profiles = json.load(f)
random.seed(42)
sample = random.sample(val_profiles, 25)
print(f"Sampled {len(sample)} validation profiles\n")

# ── VARIANT B ──────────────────────────────────────────────────────────────
print("=" * 60)
print("TUNING VARIANT B")
print("=" * 60)

results_b = []
for method in RETRIEVAL_METHODS:
    for k in K_VALUES:
        print(f"\n--- Variant B | method={method} | k={k} ---")
        scores = []
        for i, profile in enumerate(sample):
            bump_start = profile["birth_year"] + 15
            bump_end   = profile["birth_year"] + 25
            try:
                retrieved = retrieve(
                    f"popular songs {bump_start}-{bump_end}",
                    faiss_index, df, k=k, method=method, bm25_index=bm25
                )
                result = generate_playlist(profile, retrieved, GENERATION_PROMPT)
                hist   = historical_plausibility(result["playlist"], df, bump_start, bump_end)
                judge  = llm_judge(profile, result["playlist"])
                scores.append({
                    "hist_plausibility":  hist,
                    "bio_precision_llm":  judge["biographical_precision"],
                    "cultural_approp_llm": judge["cultural_appropriateness"],
                    "overall_llm":        judge["overall_quality"],
                })
                print(f"  [{i+1}/25] {profile['name']}: hist={hist:.2f}, overall={judge['overall_quality']}")
            except Exception as e:
                print(f"  [{i+1}/25] {profile['name']}: SKIPPED — {e}")

        if scores:
            avg = pd.DataFrame(scores).mean().to_dict()
            results_b.append({"variant": "B", "method": method, "k": k,
                               **{f"avg_{k2}": v for k2, v in avg.items()}})
            print(f"  >>> Avg: {avg}")

tuning_b = pd.DataFrame(results_b)
tuning_b.to_csv("outputs/tuning_variant_b.csv", index=False)
print("\nVariant B saved to outputs/tuning_variant_b.csv")

# ── VARIANT C ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TUNING VARIANT C")
print("=" * 60)

results_c = []
for method in RETRIEVAL_METHODS:
    for k in K_VALUES:
        print(f"\n--- Variant C | method={method} | total_k={k} ---")
        scores = []
        for i, profile in enumerate(sample):
            bump_start = profile["birth_year"] + 15
            bump_end   = profile["birth_year"] + 25
            try:
                retrieved, _ = profile_to_context(
                    profile, faiss_index, bm25, df,
                    method=method, k_per_query=max(k // 5, 5), total_k=k
                )
                result = generate_playlist(profile, retrieved, GENERATION_PROMPT)
                hist   = historical_plausibility(result["playlist"], df, bump_start, bump_end)
                judge  = llm_judge(profile, result["playlist"])
                scores.append({
                    "hist_plausibility":  hist,
                    "bio_precision_llm":  judge["biographical_precision"],
                    "cultural_approp_llm": judge["cultural_appropriateness"],
                    "overall_llm":        judge["overall_quality"],
                })
                print(f"  [{i+1}/25] {profile['name']}: hist={hist:.2f}, overall={judge['overall_quality']}")
            except Exception as e:
                print(f"  [{i+1}/25] {profile['name']}: SKIPPED — {e}")

        if scores:
            avg = pd.DataFrame(scores).mean().to_dict()
            results_c.append({"variant": "C", "method": method, "k": k,
                               **{f"avg_{k2}": v for k2, v in avg.items()}})
            print(f"  >>> Avg: {avg}")

tuning_c = pd.DataFrame(results_c)
tuning_c.to_csv("outputs/tuning_variant_c.csv", index=False)
print("\nVariant C saved to outputs/tuning_variant_c.csv")

# ── BEST CONFIGS ───────────────────────────────────────────────────────────
best_b = tuning_b.sort_values("avg_overall_llm", ascending=False).iloc[0]
best_c = tuning_c.sort_values("avg_overall_llm", ascending=False).iloc[0]

best_configs = {
    "variant_b": {"method": best_b["method"], "k": int(best_b["k"])},
    "variant_c": {"method": best_c["method"], "k": int(best_c["k"])},
}
with open("outputs/best_configs.json", "w") as f:
    json.dump(best_configs, f, indent=2)

print(f"\nBest Variant B: method={best_b['method']}, k={int(best_b['k'])}, avg_overall={best_b['avg_overall_llm']:.3f}")
print(f"Best Variant C: method={best_c['method']}, k={int(best_c['k'])}, avg_overall={best_c['avg_overall_llm']:.3f}")
print("\nSaved outputs/best_configs.json")
print(json.dumps(best_configs, indent=2))
