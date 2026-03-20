"""
Phase 5: Run All Three Variants
  A - Baseline LLM (no retrieval)
  B - RAG with birth-year-only filtering
  C - RAG with full biographical context (full pipeline)

Usage:
    python run_phase5.py

Requires:
    - OPENAI_API_KEY set in environment (or .env file)
    - data/knowledge_base.csv
    - data/songs.index
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from src.retrieval import load_retrieval_system, retrieve
from src.generation import generate_playlist
from src.profiling import profile_to_context
from configs.prompts import GENERATION_PROMPT

client = OpenAI()

# ---------------------------------------------------------------------------
# Variant A: Baseline prompt (no retrieval)
# ---------------------------------------------------------------------------

BASELINE_PROMPT = """You are a music therapist creating a personalized playlist for a dementia patient.

PATIENT PROFILE:
{profile}

Based on your knowledge of music history, create a ranked playlist of exactly 10 songs 
and 3 caregiver conversation cards. Focus on the patient's reminiscence bump (ages 15–25),
their cultural background, and their geographic region.

Respond in this exact JSON format:
{{
    "playlist": [
        {{
            "rank": 1,
            "song": "Song Title",
            "artist": "Artist Name",
            "year": 1965,
            "relevance": "Why this song matters for this specific patient"
        }}
    ],
    "caregiver_cards": [
        {{
            "song": "Song Title",
            "prompt": "A gentle question or statement linking this song to the patient's life"
        }}
    ]
}}
"""


def run_variant_a(profile):
    """Baseline: no retrieval, just the LLM."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a music therapist. Respond with valid JSON only."},
            {"role": "user", "content": BASELINE_PROMPT.format(
                profile=json.dumps(profile, indent=2)
            )},
        ],
        temperature=0.4,
    )
    text = response.choices[0].message.content
    text = text.strip().strip("```json").strip("```").strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Variant B: RAG with birth-year-only filtering
# ---------------------------------------------------------------------------

def run_variant_b(profile, index, df):
    """RAG with birth-year-only filtering."""
    bump_start = profile["birth_year"] + 15
    bump_end = profile["birth_year"] + 25
    query = f"popular songs {bump_start}-{bump_end}"

    retrieved = retrieve(query, index, df, k=20)
    result = generate_playlist(profile, retrieved, GENERATION_PROMPT)
    return result, retrieved


# ---------------------------------------------------------------------------
# Variant C: RAG with full biographical context
# ---------------------------------------------------------------------------

def run_variant_c(profile, index, df):
    """RAG with full biographical profiling (the full pipeline)."""
    retrieved, queries = profile_to_context(profile, index, df)
    result = generate_playlist(profile, retrieved, GENERATION_PROMPT)
    return result, retrieved, queries


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_playlist(result, variant_name):
    print(f"\n{'='*60}")
    print(f"  {variant_name}")
    print(f"{'='*60}")
    print("\n--- PLAYLIST ---")
    for song in result["playlist"]:
        print(f"  {song['rank']}. {song['song']} — {song['artist']} ({song['year']})")
        print(f"     {song['relevance']}\n")
    print("--- CAREGIVER CARDS ---")
    for card in result["caregiver_cards"]:
        print(f"  Song: {card['song']}")
        print(f"  Prompt: {card['prompt']}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample patient profile
    patient = {
        "name": "James Wilson",
        "birth_year": 1948,
        "hometown": "Detroit, Michigan",
        "cultural_background": "African American",
        "life_events": [
            {"year": 1966, "event": "Graduated from Cass Tech High School"},
            {"year": 1968, "event": "Drafted into the Vietnam War"},
            {"year": 1971, "event": "Married Dorothy in Detroit"},
            {"year": 1975, "event": "First child born"},
        ],
    }

    # Paths to pre-built data (from Phase 3 / 02_data_pipeline + 03_retrieval notebooks)
    INDEX_PATH = "data/index/songs.index"
    KB_PATH = "data/index/knowledge_base.csv"

    outputs = {}

    # --- Variant A ---
    print("\nRunning Variant A (Baseline LLM)...")
    result_a = run_variant_a(patient)
    print_playlist(result_a, "VARIANT A — Baseline LLM (no retrieval)")
    outputs["variant_a"] = result_a

    # --- Variants B & C require the FAISS index ---
    if not os.path.exists(INDEX_PATH) or not os.path.exists(KB_PATH):
        print(f"\n[WARNING] Data files not found at {INDEX_PATH} / {KB_PATH}")
        print("  Skipping Variants B and C.")
        print("  Download knowledge_base.csv and songs.index from Google Drive into data/")
    else:
        index, df = load_retrieval_system(INDEX_PATH, KB_PATH)

        # --- Variant B ---
        print("\nRunning Variant B (RAG, birth-year query)...")
        result_b, retrieved_b = run_variant_b(patient, index, df)
        print_playlist(result_b, "VARIANT B — RAG (birth-year only)")
        outputs["variant_b"] = result_b

        # --- Variant C ---
        print("\nRunning Variant C (RAG, full biographical context)...")
        result_c, retrieved_c, queries_c = run_variant_c(patient, index, df)
        print(f"  Queries used: {queries_c}")
        print_playlist(result_c, "VARIANT C — RAG (full biographical context)")
        outputs["variant_c"] = result_c

    # Save all outputs
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/phase5_results.json"
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"\nResults saved to {out_path}")
