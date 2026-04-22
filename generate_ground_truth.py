"""
Generate ground truth playlists for all profiles.
- GPT-4o generates 10 songs (reminiscence bump + cultural/regional focus)
- Claude generates 10 songs (life events + genre diversity focus)
- Total: 20 ground truth songs per profile, all restricted to Billboard Hot 100
"""
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import json, os, time
import pandas as pd
from openai import OpenAI
import anthropic

openai_client = OpenAI()
claude_client = anthropic.Anthropic()

# Load knowledge base so we can tell both LLMs which songs are available
df_kb = pd.read_csv("data/processed/knowledge_base.csv")
# Build a compact song list string filtered by era for each profile
def get_era_songs(birth_year, n=150):
    bump_start = birth_year + 10
    bump_end   = birth_year + 35
    era_df = df_kb[
        (df_kb["year"] >= bump_start) &
        (df_kb["year"] <= bump_end)
    ][["song", "artist", "year"]].drop_duplicates().head(n)
    return "\n".join(f"- \"{row['song']}\" by {row['artist']} ({int(row['year'])})"
                     for _, row in era_df.iterrows())


GPT4O_PROMPT = """You are a music therapist building a ground truth playlist for a dementia research study.

PATIENT PROFILE:
{profile}

AVAILABLE SONGS (Billboard Hot 100 — you MUST only pick from this list):
{songs}

Select exactly 10 songs from the list above that would be most meaningful for this specific patient.
Focus on: their reminiscence bump (ages 15-25), cultural background, geographic region, and era.

Respond with a JSON array of exactly 10 objects, each with: "song", "artist", "year".
No other text."""

CLAUDE_PROMPT = """You are a music therapist building a ground truth playlist for a dementia research study.

PATIENT PROFILE:
{profile}

AVAILABLE SONGS (Billboard Hot 100 — you MUST only pick from this list):
{songs}

Another researcher already selected 10 songs focused on this patient's reminiscence bump and cultural background.
Your job is to select 10 DIFFERENT songs that complement those choices.
Focus on: their specific life events, genre diversity, and songs tied to key personal milestones.
Do NOT duplicate songs from this already-selected list:
{already_selected}

Respond with a JSON array of exactly 10 objects, each with: "song", "artist", "year".
No other text."""


def parse_json_response(text):
    text = text.strip().strip("```json").strip("```").strip()
    return json.loads(text)


def generate_gpt4o_songs(profile, era_songs):
    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a music therapist. Respond with valid JSON only."},
                    {"role": "user", "content": GPT4O_PROMPT.format(
                        profile=json.dumps(profile, indent=2),
                        songs=era_songs
                    )},
                ],
                temperature=0.3,
            )
            return parse_json_response(response.choices[0].message.content)
        except Exception as e:
            if attempt < 2:
                print(f"    GPT-4o attempt {attempt+1} failed: {e}, retrying...")
                time.sleep(2)
            else:
                raise


def generate_claude_songs(profile, era_songs, already_selected):
    already_str = "\n".join(f"- \"{s['song']}\" by {s['artist']}" for s in already_selected)
    for attempt in range(3):
        try:
            response = claude_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": CLAUDE_PROMPT.format(
                        profile=json.dumps(profile, indent=2),
                        songs=era_songs,
                        already_selected=already_str
                    )},
                ],
            )
            return parse_json_response(response.content[0].text)
        except Exception as e:
            if attempt < 2:
                print(f"    Claude attempt {attempt+1} failed: {e}, retrying...")
                time.sleep(2)
            else:
                raise


def add_ground_truth(profiles_path):
    with open(profiles_path) as f:
        profiles = json.load(f)

    updated = []
    for i, profile in enumerate(profiles):
        pid = profile.get("id", f"P{i+1}")
        print(f"[{i+1}/{len(profiles)}] {pid}: {profile['name']} (b.{profile['birth_year']})")

        era_songs = get_era_songs(profile["birth_year"])

        if not era_songs:
            print(f"  WARNING: No era songs found, skipping ground truth")
            profile["ground_truth_playlist"] = []
            updated.append(profile)
            continue

        try:
            gpt4o_songs = generate_gpt4o_songs(profile, era_songs)
            print(f"  GPT-4o: {len(gpt4o_songs)} songs")
        except Exception as e:
            print(f"  GPT-4o FAILED: {e}")
            gpt4o_songs = []

        try:
            claude_songs = generate_claude_songs(profile, era_songs, gpt4o_songs)
            print(f"  Claude: {len(claude_songs)} songs")
        except Exception as e:
            print(f"  Claude FAILED: {e}")
            claude_songs = []

        # Combine, deduplicate by song title
        combined = gpt4o_songs + claude_songs
        seen = set()
        deduped = []
        for s in combined:
            key = s["song"].lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(s)

        profile["ground_truth_playlist"] = deduped
        print(f"  Total ground truth: {len(deduped)} songs")
        updated.append(profile)

    with open(profiles_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"\nSaved {profiles_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating ground truth for VAL profiles...")
    print("=" * 60)
    add_ground_truth("data/processed/val_profiles.json")

    print("\n" + "=" * 60)
    print("Generating ground truth for TEST profiles...")
    print("=" * 60)
    add_ground_truth("data/processed/test_profiles.json")

    print("\nDone. Ground truth added to all profiles.")
