"""
Phase 9: Final Evaluation on Test Set
Run after Phase 8 tuning is complete and best_configs.json exists.
"""
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import json, time, os
import pandas as pd
from src.retrieval import load_retrieval_system, retrieve
from src.profiling import profile_to_context
from src.generation import run_variant_a, run_variant_b, run_variant_c
from src.evaluation import historical_plausibility, llm_judge, create_human_eval_sheet
from configs.prompts import GENERATION_PROMPT, BASELINE_PROMPT

os.makedirs("outputs", exist_ok=True)

# --- Load system ---
print("Loading retrieval system...")
faiss_index, df, bm25 = load_retrieval_system(
    "data/index/songs.index",
    "data/processed/knowledge_base.csv",
    "data/index/bm25.pkl"
)
print(f"Loaded {faiss_index.ntotal} vectors, {len(df)} songs")

# --- Load test profiles and best configs ---
with open("data/processed/test_profiles.json") as f:
    test_profiles = json.load(f)
with open("outputs/best_configs.json") as f:
    best_configs = json.load(f)

print(f"\nTest profiles: {len(test_profiles)}")
print(f"Best configs: {json.dumps(best_configs, indent=2)}\n")

# --- Run all 3 variants on every test profile ---
results_all = {}

for profile in test_profiles:
    pid = profile["id"]
    print(f"\n{'='*50}")
    print(f"Processing {pid}: {profile['name']} ({profile['gender']}, b.{profile['birth_year']})")
    print(f"{'='*50}")

    bump_start = profile["birth_year"] + 15
    bump_end   = profile["birth_year"] + 25
    results_all[pid] = {"profile": profile}

    # Variant A
    print("  Running Variant A (baseline)...")
    start = time.time()
    result_a = run_variant_a(profile, BASELINE_PROMPT)
    time_a = time.time() - start
    hist_a = historical_plausibility(result_a["playlist"], df, bump_start, bump_end)
    judge_a = llm_judge(profile, result_a["playlist"])
    results_all[pid]["variant_a"] = {
        "result": result_a, "time": round(time_a, 1),
        "historical_plausibility": hist_a, "llm_judge": judge_a,
    }
    print(f"    hist={hist_a:.2f}, bio={judge_a['biographical_precision']}, overall={judge_a['overall_quality']}")

    # Variant B
    cfg_b = best_configs["variant_b"]
    print(f"  Running Variant B (method={cfg_b['method']}, k={cfg_b['k']})...")
    start = time.time()
    result_b, retrieved_b = run_variant_b(
        profile, faiss_index, bm25, df, GENERATION_PROMPT,
        method=cfg_b["method"], k=cfg_b["k"]
    )
    time_b = time.time() - start
    hist_b = historical_plausibility(result_b["playlist"], df, bump_start, bump_end)
    judge_b = llm_judge(profile, result_b["playlist"])
    results_all[pid]["variant_b"] = {
        "result": result_b, "time": round(time_b, 1),
        "historical_plausibility": hist_b, "llm_judge": judge_b,
        "retrieved": retrieved_b.to_dict(),
    }
    print(f"    hist={hist_b:.2f}, bio={judge_b['biographical_precision']}, overall={judge_b['overall_quality']}")

    # Variant C
    cfg_c = best_configs["variant_c"]
    print(f"  Running Variant C (method={cfg_c['method']}, k={cfg_c['k']})...")
    start = time.time()
    result_c, retrieved_c, queries_c = run_variant_c(
        profile, faiss_index, bm25, df, GENERATION_PROMPT,
        method=cfg_c["method"],
        k_per_query=max(cfg_c["k"] // 5, 5),
        total_k=cfg_c["k"]
    )
    time_c = time.time() - start
    hist_c = historical_plausibility(result_c["playlist"], df, bump_start, bump_end)
    judge_c = llm_judge(profile, result_c["playlist"])
    results_all[pid]["variant_c"] = {
        "result": result_c, "time": round(time_c, 1),
        "historical_plausibility": hist_c, "llm_judge": judge_c,
        "retrieved": retrieved_c.to_dict(), "queries": queries_c,
    }
    print(f"    hist={hist_c:.2f}, bio={judge_c['biographical_precision']}, overall={judge_c['overall_quality']}")

    # Save progress after each profile
    with open("outputs/test_experiment_results.json", "w") as f:
        json.dump(results_all, f, indent=2, default=str)

print("\nAll test profiles done. Saved outputs/test_experiment_results.json")

# --- Human eval sheet ---
create_human_eval_sheet(results_all, "outputs/human_eval_test_set.csv", set_name="test")

# --- Ablation study on Variant C ---
print("\n" + "="*60)
print("ABLATION STUDY")
print("="*60)

def run_ablation(profile, remove_field, method, total_k):
    ablated = profile.copy()
    if remove_field == "region":
        ablated["hometown"] = "United States"
    elif remove_field == "life_events":
        ablated["life_events"] = []
    elif remove_field == "culture":
        ablated["cultural_background"] = "American"
    elif remove_field == "gender":
        ablated.pop("gender", None)
    retrieved, _ = profile_to_context(
        ablated, faiss_index, bm25, df,
        method=method, k_per_query=max(total_k // 5, 5), total_k=total_k
    )
    from src.generation import generate_playlist
    return generate_playlist(ablated, retrieved, GENERATION_PROMPT)

cfg_c = best_configs["variant_c"]
ablation_results = []

for field in ["region", "life_events", "culture", "gender"]:
    print(f"\n--- Ablation: removing {field} ---")
    for profile in test_profiles:
        result = run_ablation(profile, field, method=cfg_c["method"], total_k=cfg_c["k"])
        judge = llm_judge(profile, result["playlist"])
        ablation_results.append({
            "profile_id": profile["id"],
            "removed_field": field,
            **judge,
        })
        print(f"  {profile['id']}: overall={judge['overall_quality']}")

ablation_df = pd.DataFrame(ablation_results)
print("\nAblation summary:")
print(ablation_df.groupby("removed_field")[
    ["biographical_precision", "cultural_appropriateness", "overall_quality"]
].mean().round(3))
ablation_df.to_csv("outputs/ablation_results.csv", index=False)
print("\nSaved outputs/ablation_results.csv")
