"""
Phase 8: Hyperparameter Tuning on Validation Set
Tunes Variant B and Variant C separately over:
  - methods: dense, bm25
  - k values: 10, 20, 30
Run: python3 -u run_phase8.py
"""
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import json, random, os
import pandas as pd
from src.retrieval import load_retrieval_system, retrieve
from src.profiling import profile_to_context
from src.generation import generate_playlist
from src.evaluation import historical_plausibility, llm_judge, precision_at_k, recall, mrr
from configs.prompts import GENERATION_PROMPT

RETRIEVAL_METHODS = ["dense", "bm25"]
K_VALUES = [10, 20, 30]

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


def get_ground_truth_songs(profile):
    """Return list of song titles from the profile's ground truth playlist."""
    return [s["song"] for s in profile.get("ground_truth_playlist", [])]


def load_checkpoint(path):
    """Load existing raw results as a set of completed (variant, method, k, profile_id) keys."""
    if not os.path.exists(path):
        return pd.DataFrame(), set()
    existing = pd.read_csv(path)
    done = set(zip(existing["variant"], existing["method"], existing["k"], existing["profile_id"]))
    print(f"  Resuming: found {len(done)} already-completed profiles in {path}")
    return existing, done


# ── VARIANT B ──────────────────────────────────────────────────────────────
print("=" * 60)
print("TUNING VARIANT B")
print("=" * 60)

existing_b, done_b = load_checkpoint("outputs/tuning_variant_b_raw.csv")
existing_playlists_b, _ = load_checkpoint("outputs/tuning_variant_b_playlists.csv")
raw_b = existing_b.to_dict("records") if not existing_b.empty else []
playlists_b = existing_playlists_b.to_dict("records") if not existing_playlists_b.empty else []
results_b = []

for method in RETRIEVAL_METHODS:
    for k in K_VALUES:
        print(f"\n--- Variant B | method={method} | k={k} ---")
        scores = []
        for i, profile in enumerate(sample):
            pid = profile.get("id")
            if ("B", method, k, pid) in done_b:
                print(f"  [{i+1}/25] {profile['name']}: SKIPPED (already done)")
                existing_row = existing_b[
                    (existing_b["method"] == method) &
                    (existing_b["k"] == k) &
                    (existing_b["profile_id"] == pid)
                ]
                if not existing_row.empty:
                    scores.append(existing_row.iloc[0].to_dict())
                continue
            bump_start = profile["birth_year"] + 15
            bump_end   = profile["birth_year"] + 25
            gt_songs   = get_ground_truth_songs(profile)
            try:
                retrieved = retrieve(
                    f"popular songs {bump_start}-{bump_end}",
                    faiss_index, df, k=k, method=method, bm25_index=bm25
                )
                result = generate_playlist(profile, retrieved, GENERATION_PROMPT)
                hist   = historical_plausibility(result["playlist"], df, bump_start, bump_end)
                judge  = llm_judge(profile, result["playlist"], ground_truth=profile.get("ground_truth_playlist"))
                p_at_k = precision_at_k(retrieved, gt_songs)
                rec    = recall(retrieved, gt_songs)
                rr     = mrr(retrieved, gt_songs)
                row = {
                    "variant": "B", "method": method, "k": k,
                    "profile_id": profile.get("id"), "name": profile["name"],
                    "birth_year": profile["birth_year"], "cultural_background": profile["cultural_background"],
                    "hist_plausibility":   hist,
                    "bio_precision_llm":   judge["biographical_precision"],
                    "cultural_approp_llm": judge["cultural_appropriateness"],
                    "overall_llm":         judge["overall_quality"],
                    "precision_at_k":      p_at_k,
                    "recall":              rec,
                    "mrr":                 rr,
                }
                scores.append(row)
                raw_b.append(row)
                # Save full playlist for human eval
                life_events_str = " | ".join(
                    f"{e['year']}: {e['event']}" for e in profile.get("life_events", [])
                )
                for song in result["playlist"]:
                    playlists_b.append({
                        "condition": f"B_{method}_k{k}",
                        "variant": "B", "method": method, "k": k,
                        "profile_id": profile.get("id"), "name": profile["name"],
                        "gender": profile["gender"], "birth_year": profile["birth_year"],
                        "cultural_background": profile["cultural_background"],
                        "hometown": profile["hometown"],
                        "life_events": life_events_str,
                        "rank": song["rank"], "song": song["song"],
                        "artist": song["artist"], "year": song["year"],
                        "relevance_reason": song.get("relevance", ""),
                        "biographical_precision_1_5": "",
                        "cultural_appropriateness_1_5": "",
                        "notes": "",
                    })
                # Save checkpoint after every profile
                pd.DataFrame(raw_b).to_csv("outputs/tuning_variant_b_raw.csv", index=False)
                pd.DataFrame(playlists_b).to_csv("outputs/tuning_variant_b_playlists.csv", index=False)
                print(f"  [{i+1}/25] {profile['name']}: hist={hist:.2f}, overall={judge['overall_quality']}, P@k={p_at_k:.2f}, MRR={rr:.2f}")
            except Exception as e:
                print(f"  [{i+1}/25] {profile['name']}: SKIPPED — {e}")

        if scores:
            avg = pd.DataFrame(scores).mean(numeric_only=True).to_dict()
            results_b.append({"variant": "B", "method": method, "k": k,
                               **{f"avg_{k2}": v for k2, v in avg.items()}})
            print(f"  >>> Avg: {avg}")

tuning_b = pd.DataFrame(results_b)
tuning_b.to_csv("outputs/tuning_variant_b.csv", index=False)
pd.DataFrame(raw_b).to_csv("outputs/tuning_variant_b_raw.csv", index=False)
pd.DataFrame(playlists_b).to_csv("outputs/tuning_variant_b_playlists.csv", index=False)
print("\nVariant B saved to outputs/tuning_variant_b.csv, tuning_variant_b_raw.csv, tuning_variant_b_playlists.csv")

# ── VARIANT C ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TUNING VARIANT C")
print("=" * 60)

existing_c, done_c = load_checkpoint("outputs/tuning_variant_c_raw.csv")
existing_playlists_c, _ = load_checkpoint("outputs/tuning_variant_c_playlists.csv")
raw_c = existing_c.to_dict("records") if not existing_c.empty else []
playlists_c = existing_playlists_c.to_dict("records") if not existing_playlists_c.empty else []
results_c = []

for method in RETRIEVAL_METHODS:
    for k in K_VALUES:
        print(f"\n--- Variant C | method={method} | total_k={k} ---")
        scores = []
        for i, profile in enumerate(sample):
            pid = profile.get("id")
            if ("C", method, k, pid) in done_c:
                print(f"  [{i+1}/25] {profile['name']}: SKIPPED (already done)")
                existing_row = existing_c[
                    (existing_c["method"] == method) &
                    (existing_c["k"] == k) &
                    (existing_c["profile_id"] == pid)
                ]
                if not existing_row.empty:
                    scores.append(existing_row.iloc[0].to_dict())
                continue
            bump_start = profile["birth_year"] + 15
            bump_end   = profile["birth_year"] + 25
            gt_songs   = get_ground_truth_songs(profile)
            try:
                retrieved, _ = profile_to_context(
                    profile, faiss_index, bm25, df,
                    method=method, k_per_query=max(k // 5, 5), total_k=k
                )
                result = generate_playlist(profile, retrieved, GENERATION_PROMPT)
                hist   = historical_plausibility(result["playlist"], df, bump_start, bump_end)
                judge  = llm_judge(profile, result["playlist"], ground_truth=profile.get("ground_truth_playlist"))
                p_at_k = precision_at_k(retrieved, gt_songs)
                rec    = recall(retrieved, gt_songs)
                rr     = mrr(retrieved, gt_songs)
                row = {
                    "variant": "C", "method": method, "k": k,
                    "profile_id": profile.get("id"), "name": profile["name"],
                    "birth_year": profile["birth_year"], "cultural_background": profile["cultural_background"],
                    "hist_plausibility":   hist,
                    "bio_precision_llm":   judge["biographical_precision"],
                    "cultural_approp_llm": judge["cultural_appropriateness"],
                    "overall_llm":         judge["overall_quality"],
                    "precision_at_k":      p_at_k,
                    "recall":              rec,
                    "mrr":                 rr,
                }
                scores.append(row)
                raw_c.append(row)
                # Save full playlist for human eval
                life_events_str = " | ".join(
                    f"{e['year']}: {e['event']}" for e in profile.get("life_events", [])
                )
                for song in result["playlist"]:
                    playlists_c.append({
                        "condition": f"C_{method}_k{k}",
                        "variant": "C", "method": method, "k": k,
                        "profile_id": profile.get("id"), "name": profile["name"],
                        "gender": profile["gender"], "birth_year": profile["birth_year"],
                        "cultural_background": profile["cultural_background"],
                        "hometown": profile["hometown"],
                        "life_events": life_events_str,
                        "rank": song["rank"], "song": song["song"],
                        "artist": song["artist"], "year": song["year"],
                        "relevance_reason": song.get("relevance", ""),
                        "biographical_precision_1_5": "",
                        "cultural_appropriateness_1_5": "",
                        "notes": "",
                    })
                # Save checkpoint after every profile
                pd.DataFrame(raw_c).to_csv("outputs/tuning_variant_c_raw.csv", index=False)
                pd.DataFrame(playlists_c).to_csv("outputs/tuning_variant_c_playlists.csv", index=False)
                print(f"  [{i+1}/25] {profile['name']}: hist={hist:.2f}, overall={judge['overall_quality']}, P@k={p_at_k:.2f}, MRR={rr:.2f}")
            except Exception as e:
                print(f"  [{i+1}/25] {profile['name']}: SKIPPED — {e}")

        if scores:
            avg = pd.DataFrame(scores).mean(numeric_only=True).to_dict()
            results_c.append({"variant": "C", "method": method, "k": k,
                               **{f"avg_{k2}": v for k2, v in avg.items()}})
            print(f"  >>> Avg: {avg}")

tuning_c = pd.DataFrame(results_c)
tuning_c.to_csv("outputs/tuning_variant_c.csv", index=False)
pd.DataFrame(raw_c).to_csv("outputs/tuning_variant_c_raw.csv", index=False)
pd.DataFrame(playlists_c).to_csv("outputs/tuning_variant_c_playlists.csv", index=False)
print("\nVariant C saved to outputs/tuning_variant_c.csv, tuning_variant_c_raw.csv, tuning_variant_c_playlists.csv")

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
