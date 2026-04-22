"""
Phase 9: Final Evaluation on Test Set
Run after Phase 8 tuning is complete and best_configs.json exists.
Run: python3 -u run_phase9.py | tee outputs/phase9_log.txt
"""
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import json, time, os
import pandas as pd
from src.retrieval import load_retrieval_system, retrieve
from src.profiling import profile_to_context
from src.generation import run_variant_a, run_variant_b, run_variant_c
from src.evaluation import (
    historical_plausibility, llm_judge, create_human_eval_sheet,
    precision_at_k, recall, mrr
)
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
raw_rows = []      # flat per-profile per-variant scores for CSV
playlist_rows = [] # song-level rows for human eval

for profile in test_profiles:
    pid = profile["id"]
    gt_songs = [s["song"] for s in profile.get("ground_truth_playlist", [])]
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
    hist_a  = historical_plausibility(result_a["playlist"], df, bump_start, bump_end)
    judge_a = llm_judge(profile, result_a["playlist"], ground_truth=profile.get("ground_truth_playlist"))
    results_all[pid]["variant_a"] = {
        "result": result_a, "time": round(time_a, 1),
        "historical_plausibility": hist_a, "llm_judge": judge_a,
    }
    raw_rows.append({
        "profile_id": pid, "name": profile["name"], "gender": profile["gender"],
        "birth_year": profile["birth_year"], "cultural_background": profile["cultural_background"],
        "variant": "A", "method": "none", "k": "none",
        "hist_plausibility": hist_a,
        "bio_precision_llm": judge_a["biographical_precision"],
        "cultural_approp_llm": judge_a["cultural_appropriateness"],
        "overall_llm": judge_a["overall_quality"],
        "precision_at_k": None, "recall": None, "mrr": None,
        "time_sec": round(time_a, 1),
    })
    life_events_str = " | ".join(
        f"{e['year']}: {e['event']}" for e in profile.get("life_events", [])
    )
    for song in result_a["playlist"]:
        playlist_rows.append({
            "condition": "A_none_knone",
            "variant": "A", "method": "none", "k": "none",
            "profile_id": pid, "name": profile["name"], "gender": profile["gender"],
            "birth_year": profile["birth_year"], "cultural_background": profile["cultural_background"],
            "hometown": profile["hometown"], "life_events": life_events_str,
            "rank": song["rank"], "song": song["song"], "artist": song["artist"], "year": song["year"],
            "relevance_reason": song.get("relevance", ""),
            "biographical_precision_1_5": "", "cultural_appropriateness_1_5": "", "notes": "",
        })
    print(f"    hist={hist_a:.2f}, bio={judge_a['biographical_precision']}, overall={judge_a['overall_quality']}")

    # Variant B
    cfg_b = best_configs["variant_b"]
    print(f"  Running Variant B (method={cfg_b['method']}, k={cfg_b['k']})...")
    start = time.time()
    result_b, retrieved_b = run_variant_b(
        profile, faiss_index, bm25, df, GENERATION_PROMPT,
        method=cfg_b["method"], k=cfg_b["k"]
    )
    time_b  = time.time() - start
    hist_b  = historical_plausibility(result_b["playlist"], df, bump_start, bump_end)
    judge_b = llm_judge(profile, result_b["playlist"], ground_truth=profile.get("ground_truth_playlist"))
    p_at_k_b = precision_at_k(retrieved_b, gt_songs)
    rec_b    = recall(retrieved_b, gt_songs)
    mrr_b    = mrr(retrieved_b, gt_songs)
    results_all[pid]["variant_b"] = {
        "result": result_b, "time": round(time_b, 1),
        "historical_plausibility": hist_b, "llm_judge": judge_b,
        "precision_at_k": p_at_k_b, "recall": rec_b, "mrr": mrr_b,
        "retrieved": retrieved_b.to_dict(),
    }
    raw_rows.append({
        "profile_id": pid, "name": profile["name"], "gender": profile["gender"],
        "birth_year": profile["birth_year"], "cultural_background": profile["cultural_background"],
        "variant": "B", "method": cfg_b["method"], "k": cfg_b["k"],
        "hist_plausibility": hist_b,
        "bio_precision_llm": judge_b["biographical_precision"],
        "cultural_approp_llm": judge_b["cultural_appropriateness"],
        "overall_llm": judge_b["overall_quality"],
        "precision_at_k": p_at_k_b, "recall": rec_b, "mrr": mrr_b,
        "time_sec": round(time_b, 1),
    })
    for song in result_b["playlist"]:
        playlist_rows.append({
            "condition": f"B_{cfg_b['method']}_k{cfg_b['k']}",
            "variant": "B", "method": cfg_b["method"], "k": cfg_b["k"],
            "profile_id": pid, "name": profile["name"], "gender": profile["gender"],
            "birth_year": profile["birth_year"], "cultural_background": profile["cultural_background"],
            "hometown": profile["hometown"], "life_events": life_events_str,
            "rank": song["rank"], "song": song["song"], "artist": song["artist"], "year": song["year"],
            "relevance_reason": song.get("relevance", ""),
            "biographical_precision_1_5": "", "cultural_appropriateness_1_5": "", "notes": "",
        })
    print(f"    hist={hist_b:.2f}, bio={judge_b['biographical_precision']}, overall={judge_b['overall_quality']}, P@k={p_at_k_b:.2f}, MRR={mrr_b:.2f}")

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
    time_c  = time.time() - start
    hist_c  = historical_plausibility(result_c["playlist"], df, bump_start, bump_end)
    judge_c = llm_judge(profile, result_c["playlist"], ground_truth=profile.get("ground_truth_playlist"))
    p_at_k_c = precision_at_k(retrieved_c, gt_songs)
    rec_c    = recall(retrieved_c, gt_songs)
    mrr_c    = mrr(retrieved_c, gt_songs)
    results_all[pid]["variant_c"] = {
        "result": result_c, "time": round(time_c, 1),
        "historical_plausibility": hist_c, "llm_judge": judge_c,
        "precision_at_k": p_at_k_c, "recall": rec_c, "mrr": mrr_c,
        "retrieved": retrieved_c.to_dict(), "queries": queries_c,
    }
    raw_rows.append({
        "profile_id": pid, "name": profile["name"], "gender": profile["gender"],
        "birth_year": profile["birth_year"], "cultural_background": profile["cultural_background"],
        "variant": "C", "method": cfg_c["method"], "k": cfg_c["k"],
        "hist_plausibility": hist_c,
        "bio_precision_llm": judge_c["biographical_precision"],
        "cultural_approp_llm": judge_c["cultural_appropriateness"],
        "overall_llm": judge_c["overall_quality"],
        "precision_at_k": p_at_k_c, "recall": rec_c, "mrr": mrr_c,
        "time_sec": round(time_c, 1),
    })
    for song in result_c["playlist"]:
        playlist_rows.append({
            "condition": f"C_{cfg_c['method']}_k{cfg_c['k']}",
            "variant": "C", "method": cfg_c["method"], "k": cfg_c["k"],
            "profile_id": pid, "name": profile["name"], "gender": profile["gender"],
            "birth_year": profile["birth_year"], "cultural_background": profile["cultural_background"],
            "hometown": profile["hometown"], "life_events": life_events_str,
            "rank": song["rank"], "song": song["song"], "artist": song["artist"], "year": song["year"],
            "relevance_reason": song.get("relevance", ""),
            "biographical_precision_1_5": "", "cultural_appropriateness_1_5": "", "notes": "",
        })
    print(f"    hist={hist_c:.2f}, bio={judge_c['biographical_precision']}, overall={judge_c['overall_quality']}, P@k={p_at_k_c:.2f}, MRR={mrr_c:.2f}")

    # Save progress after each profile (JSON + CSVs)
    with open("outputs/test_experiment_results.json", "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    pd.DataFrame(raw_rows).to_csv("outputs/test_results_raw.csv", index=False)
    pd.DataFrame(playlist_rows).to_csv("outputs/test_playlists.csv", index=False)

print("\nAll test profiles done.")
print("Saved outputs/test_experiment_results.json")
print("Saved outputs/test_results_raw.csv")
print("Saved outputs/test_playlists.csv")

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
        try:
            result = run_ablation(profile, field, method=cfg_c["method"], total_k=cfg_c["k"])
            judge  = llm_judge(profile, result["playlist"])
            ablation_results.append({
                "profile_id": profile["id"], "name": profile["name"],
                "gender": profile["gender"], "birth_year": profile["birth_year"],
                "cultural_background": profile["cultural_background"],
                "removed_field": field,
                "bio_precision_llm": judge["biographical_precision"],
                "cultural_approp_llm": judge["cultural_appropriateness"],
                "overall_llm": judge["overall_quality"],
            })
            print(f"  {profile['id']} {profile['name']}: overall={judge['overall_quality']}")
        except Exception as e:
            print(f"  {profile['id']}: SKIPPED — {e}")

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv("outputs/ablation_results.csv", index=False)
print("\nAblation summary:")
print(ablation_df.groupby("removed_field")[
    ["bio_precision_llm", "cultural_approp_llm", "overall_llm"]
].mean().round(3))
print("\nSaved outputs/ablation_results.csv")
