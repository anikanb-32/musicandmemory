"""
Human Evaluation Setup
Run after Phase 9 to generate rating sheets for 2 raters.
Run: python3 run_human_eval.py

This creates:
  outputs/human_eval_rater1.csv  — for rater 1 to fill in
  outputs/human_eval_rater2.csv  — for rater 2 to fill in (same profiles, different order)

After both raters fill in their sheets, run:
  python3 run_human_eval.py --compute-kappa
to compute Cohen's Kappa inter-rater agreement.
"""
import sys
import json
import random
import pandas as pd

N_PROFILES = 30  # number of profiles to rate

# ── GENERATE RATING SHEETS ─────────────────────────────────────────────────
if "--compute-kappa" not in sys.argv:

    with open("outputs/test_experiment_results.json") as f:
        results = json.load(f)

    # Sample 30 profiles reproducibly
    random.seed(42)
    all_pids = list(results.keys())
    selected_pids = random.sample(all_pids, min(N_PROFILES, len(all_pids)))
    print(f"Selected {len(selected_pids)} profiles for human evaluation")

    rows = []
    for pid in selected_pids:
        data = results[pid]
        profile = data["profile"]

        for variant_key, variant_label in [("variant_a", "A"), ("variant_b", "B"), ("variant_c", "C")]:
            if variant_key not in data:
                continue
            playlist = data[variant_key]["result"]["playlist"]
            for song in playlist:
                rows.append({
                    "profile_id": pid,
                    "patient_name": profile["name"],
                    "gender": profile["gender"],
                    "birth_year": profile["birth_year"],
                    "cultural_background": profile["cultural_background"],
                    "hometown": profile["hometown"],
                    "variant": variant_label,
                    "rank": song["rank"],
                    "song": song["song"],
                    "artist": song["artist"],
                    "year": song["year"],
                    "relevance_reason": song.get("relevance", ""),
                    # Rater fills these in:
                    "biographical_precision_1_5": "",
                    "cultural_appropriateness_1_5": "",
                    "notes": "",
                })

    df = pd.DataFrame(rows)

    # Rater 1 — sorted by profile then variant
    df.to_csv("outputs/human_eval_rater1.csv", index=False)
    print("Saved outputs/human_eval_rater1.csv")

    # Rater 2 — same data, profiles shuffled so raters can't anchor on each other
    random.seed(99)
    shuffled_pids = selected_pids.copy()
    random.shuffle(shuffled_pids)
    df2 = pd.concat([df[df["profile_id"] == pid] for pid in shuffled_pids]).reset_index(drop=True)
    df2.to_csv("outputs/human_eval_rater2.csv", index=False)
    print("Saved outputs/human_eval_rater2.csv")

    print(f"\nTotal rows per rater: {len(df)} ({len(selected_pids)} profiles × 3 variants × 10 songs)")
    print("\nInstructions:")
    print("  Fill in 'biographical_precision_1_5' and 'cultural_appropriateness_1_5'")
    print("  Scale: 1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent")
    print("  Rate each song independently based on the patient profile shown.")

# ── COMPUTE COHEN'S KAPPA ──────────────────────────────────────────────────
else:
    from sklearn.metrics import cohen_kappa_score

    rater1 = pd.read_csv("outputs/human_eval_rater1.csv")
    rater2 = pd.read_csv("outputs/human_eval_rater2.csv")

    # Align on profile_id + variant + song
    merged = pd.merge(
        rater1[["profile_id", "variant", "song", "biographical_precision_1_5", "cultural_appropriateness_1_5"]],
        rater2[["profile_id", "variant", "song", "biographical_precision_1_5", "cultural_appropriateness_1_5"]],
        on=["profile_id", "variant", "song"],
        suffixes=("_r1", "_r2")
    ).dropna()

    print(f"Aligned {len(merged)} rated songs for kappa computation\n")

    kappa_bio = cohen_kappa_score(
        merged["biographical_precision_1_5_r1"].astype(int),
        merged["biographical_precision_1_5_r2"].astype(int),
        weights="quadratic"
    )
    kappa_culture = cohen_kappa_score(
        merged["cultural_appropriateness_1_5_r1"].astype(int),
        merged["cultural_appropriateness_1_5_r2"].astype(int),
        weights="quadratic"
    )

    print(f"Cohen's Kappa (biographical precision):    {kappa_bio:.3f}")
    print(f"Cohen's Kappa (cultural appropriateness): {kappa_culture:.3f}")
    print()
    print("Interpretation: <0.2 poor | 0.2-0.4 fair | 0.4-0.6 moderate | 0.6-0.8 substantial | >0.8 almost perfect")

    # Also compare human scores vs LLM judge
    with open("outputs/test_experiment_results.json") as f:
        results = json.load(f)

    llm_rows = []
    for pid, data in results.items():
        for variant_key, variant_label in [("variant_a", "A"), ("variant_b", "B"), ("variant_c", "C")]:
            if variant_key not in data:
                continue
            judge = data[variant_key].get("llm_judge", {})
            llm_rows.append({
                "profile_id": pid,
                "variant": variant_label,
                "llm_bio": judge.get("biographical_precision"),
                "llm_culture": judge.get("cultural_appropriateness"),
            })

    llm_df = pd.DataFrame(llm_rows)

    # Average human scores per profile+variant for comparison
    human_avg = merged.groupby(["profile_id", "variant"]).agg(
        human_bio=("biographical_precision_1_5_r1", "mean"),
        human_culture=("cultural_appropriateness_1_5_r1", "mean"),
    ).reset_index()

    comparison = pd.merge(human_avg, llm_df, on=["profile_id", "variant"])
    corr_bio    = comparison["human_bio"].corr(comparison["llm_bio"])
    corr_culture = comparison["human_culture"].corr(comparison["llm_culture"])

    print(f"\nHuman vs LLM Judge correlation:")
    print(f"  Biographical precision:    r = {corr_bio:.3f}")
    print(f"  Cultural appropriateness: r = {corr_culture:.3f}")

    comparison.to_csv("outputs/human_vs_llm_comparison.csv", index=False)
    print("\nSaved outputs/human_vs_llm_comparison.csv")
