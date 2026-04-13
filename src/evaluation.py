import json
import csv
import pandas as pd
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

client = OpenAI()

# --- Retrieval metrics ---

def precision_at_k(retrieved_df, ground_truth_songs):
    """Of the top-k retrieved songs, how many are in the ground truth?"""
    retrieved_set = set(retrieved_df["song"].str.lower())
    gt_set = set(s.lower() for s in ground_truth_songs)
    if len(retrieved_set) == 0:
        return 0.0
    return len(retrieved_set & gt_set) / len(retrieved_set)

def recall(retrieved_df, ground_truth_songs):
    """What proportion of ground truth songs were retrieved?"""
    retrieved_set = set(retrieved_df["song"].str.lower())
    gt_set = set(s.lower() for s in ground_truth_songs)
    if len(gt_set) == 0:
        return 0.0
    return len(retrieved_set & gt_set) / len(gt_set)

def mrr(retrieved_df, ground_truth_songs):
    """Mean reciprocal rank — rank of first ground-truth hit."""
    gt_set = set(s.lower() for s in ground_truth_songs)
    for rank, song in enumerate(retrieved_df["song"].str.lower(), 1):
        if song in gt_set:
            return 1.0 / rank
    return 0.0

def historical_plausibility(playlist, df_kb, bump_start, bump_end, tolerance=5):
    """Fraction of playlist songs that exist in the knowledge base within the time window."""
    hits = 0
    total = len(playlist)
    for song in playlist:
        song_title = song["song"].lower()
        match = df_kb[
            (df_kb["song"].str.lower() == song_title) &
            (df_kb["year"] >= bump_start - tolerance) &
            (df_kb["year"] <= bump_end + tolerance)
        ]
        if len(match) > 0:
            hits += 1
    return hits / total if total > 0 else 0

LLM_JUDGE_PROMPT = """You are an expert music therapist evaluating a personalized playlist
generated for a dementia patient. Rate the playlist on three dimensions (1-5 scale each).

PATIENT PROFILE:
{profile}

GENERATED PLAYLIST:
{playlist}

{ground_truth_section}

SCORING CRITERIA:
- biographical_precision (1-5): How well do the songs connect to the patient's specific
  life events, time periods, and personal history?
- cultural_appropriateness (1-5): How well do the songs match the patient's cultural
  background, geographic region, and community?
- overall_quality (1-5): Overall quality as a therapeutic playlist for this specific patient.

Respond with JSON only:
{{"biographical_precision": X, "cultural_appropriateness": X, "overall_quality": X, "reasoning": "brief explanation"}}
"""

def llm_judge(profile, playlist, ground_truth=None):
    """Use GPT-4o to score a generated playlist."""
    gt_section = ""
    if ground_truth:
        gt_section = f"GROUND TRUTH PLAYLIST (for reference):\n{json.dumps(ground_truth, indent=2)}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": LLM_JUDGE_PROMPT.format(
                profile=json.dumps(profile, indent=2),
                playlist=json.dumps(playlist, indent=2),
                ground_truth_section=gt_section,
            )},
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content
    text = text.strip().strip("```json").strip("```").strip()
    return json.loads(text)


def create_human_eval_sheet(results_all, output_path, set_name="val"):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Set", "Profile ID", "Patient Name", "Gender", "Birth Year",
            "Cultural Background", "Variant", "Song Rank",
            "Song", "Artist", "Year", "Relevance Reason",
            "Biographical Precision (1-5)", "Cultural Appropriateness (1-5)",
            "Rater"
        ])

        for pid, data in results_all.items():
            profile = data["profile"]
            for variant in ["variant_a", "variant_b", "variant_c"]:
                for song in data[variant]["result"]["playlist"]:
                    writer.writerow([
                        set_name, pid, profile["name"], profile["gender"],
                        profile["birth_year"], profile["cultural_background"],
                        variant, song["rank"], song["song"], song["artist"],
                        song["year"], song["relevance"],
                        "", "", ""
                    ])

    print(f"Created {output_path}")

def compute_inter_rater(rater1_path, rater2_path):
    rater1 = pd.read_csv(rater1_path)
    rater2 = pd.read_csv(rater2_path)

    kappa_bio = cohen_kappa_score(
        rater1["Biographical Precision (1-5)"],
        rater2["Biographical Precision (1-5)"],
        weights="quadratic"
    )
    kappa_culture = cohen_kappa_score(
        rater1["Cultural Appropriateness (1-5)"],
        rater2["Cultural Appropriateness (1-5)"],
        weights="quadratic"
    )

    print(f"Cohen's kappa (biographical precision): {kappa_bio:.3f}")
    print(f"Cohen's kappa (cultural appropriateness): {kappa_culture:.3f}")
    return kappa_bio, kappa_culture

