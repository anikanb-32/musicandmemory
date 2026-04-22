"""
Phase 10: Compile and summarize final results.
Run after Phase 9 is complete.
Run: python3 -u run_phase10.py | tee outputs/phase10_log.txt
"""
import json
import pandas as pd

# --- Load raw results ---
df = pd.read_csv("outputs/test_results_raw.csv")
ablation_df = pd.read_csv("outputs/ablation_results.csv")

print("=" * 60)
print("PHASE 10: FINAL RESULTS SUMMARY")
print("=" * 60)

# --- Overall comparison: A vs B vs C ---
print("\n--- Overall: Variant A vs B vs C ---")
overall = df.groupby("variant")[
    ["hist_plausibility", "bio_precision_llm", "cultural_approp_llm", "overall_llm",
     "precision_at_k", "recall", "mrr"]
].mean().round(3)
print(overall.to_string())
overall.to_csv("outputs/summary_overall.csv")

# --- By gender ---
print("\n--- By Gender ---")
by_gender = df.groupby(["variant", "gender"])[
    ["hist_plausibility", "overall_llm"]
].mean().round(3)
print(by_gender.to_string())
by_gender.to_csv("outputs/summary_by_gender.csv")

# --- By birth decade ---
df["decade"] = (df["birth_year"] // 10) * 10
print("\n--- By Birth Decade ---")
by_decade = df.groupby(["variant", "decade"])[
    ["hist_plausibility", "overall_llm"]
].mean().round(3)
print(by_decade.to_string())
by_decade.to_csv("outputs/summary_by_decade.csv")

# --- By cultural background (top 10 most common) ---
top_backgrounds = df["cultural_background"].value_counts().head(10).index
df_top = df[df["cultural_background"].isin(top_backgrounds)]
print("\n--- By Cultural Background (top 10) ---")
by_culture = df_top.groupby(["variant", "cultural_background"])[
    ["hist_plausibility", "overall_llm"]
].mean().round(3)
print(by_culture.to_string())
by_culture.to_csv("outputs/summary_by_culture.csv")

# --- Ablation study summary ---
print("\n--- Ablation Study (Variant C, removing one field at a time) ---")
ablation_summary = ablation_df.groupby("removed_field")[
    ["bio_precision_llm", "cultural_approp_llm", "overall_llm"]
].mean().round(3)
print(ablation_summary.to_string())

# Load full Variant C scores to show baseline for ablation
variant_c = df[df["variant"] == "C"][
    ["bio_precision_llm", "cultural_approp_llm", "overall_llm"]
].mean().round(3)
print(f"\nVariant C (full, no ablation): {variant_c.to_dict()}")

print("\n--- Files saved ---")
print("  outputs/summary_overall.csv")
print("  outputs/summary_by_gender.csv")
print("  outputs/summary_by_decade.csv")
print("  outputs/summary_by_culture.csv")
print("\nDone.")
