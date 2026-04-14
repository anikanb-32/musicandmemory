from openai import OpenAI
from dotenv import load_dotenv
import json
import random

load_dotenv()
client = OpenAI()

PROFILE_GEN_PROMPT = """Generate {n} fictional patient profiles for a music therapy study.
Each profile must have these exact fields:
- name (realistic full name)
- gender ("Male" or "Female")
- birth_year (between {year_min} and {year_max})
- hometown (a specific US city and state)
- cultural_background (one of: {backgrounds})
- life_events (list of 3-5 events, each with "year" and "event" keys)

Requirements:
- The life events should be realistic for the person's background, era, and region
- Life events should span ages 15–35
- Include a mix of milestones: education, work, marriage, children, community events, migration
- Each profile should feel distinct and specific

Respond with a JSON array of profiles only, no other text.
"""

BACKGROUNDS = [
    "African American", "Latino/Mexican American", "Korean American",
    "Filipino American", "White American", "Chinese American",
    "Puerto Rican American", "Vietnamese American", "Italian American",
    "Irish American", "Japanese American", "Indian American"
]

REGIONS = [
    "Detroit, Michigan", "Los Angeles, California", "Chicago, Illinois",
    "New York City, New York", "Houston, Texas", "Memphis, Tennessee",
    "San Francisco, California", "Atlanta, Georgia", "Miami, Florida",
    "Philadelphia, Pennsylvania", "Boston, Massachusetts", "New Orleans, Louisiana",
    "Seattle, Washington", "Cleveland, Ohio", "Newark, New Jersey",
    "Minneapolis, Minnesota", "Denver, Colorado", "Dallas, Texas"
]


def generate_profile_batch(n=10, year_min=1935, year_max=1965):
    """Generate a batch of profiles via GPT-4o."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Generate fictional patient profiles. Respond with JSON only."},
            {"role": "user", "content": PROFILE_GEN_PROMPT.format(
                n=n,
                year_min=year_min,
                year_max=year_max,
                backgrounds=", ".join(BACKGROUNDS),
            )},
        ],
        temperature=0.9,
    )
    text = response.choices[0].message.content
    text = text.strip().strip("```json").strip("```").strip()
    return json.loads(text)


def generate_all_profiles(total=220, batch_size=10):
    """Generate all profiles in batches, add IDs."""
    all_profiles = []
    batches_needed = (total + batch_size - 1) // batch_size

    for b in range(batches_needed):
        print(f"Generating batch {b+1}/{batches_needed}...")
        batch = generate_profile_batch(n=batch_size)
        all_profiles.extend(batch)

    # Trim to exact count and add IDs
    all_profiles = all_profiles[:total]
    for i, p in enumerate(all_profiles):
        p["id"] = f"P{i+1:03d}"

    return all_profiles


if __name__ == "__main__":
    import os
    from collections import Counter

    profiles = generate_all_profiles(total=220)

    # Verify gender field exists on all
    assert all("gender" in p for p in profiles), "Some profiles missing gender!"

    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(profiles)
    split_idx = int(len(profiles) * 0.8)
    val_profiles = profiles[:split_idx]
    test_profiles = profiles[split_idx:]

    print(f"\nTotal: {len(profiles)} | Validation: {len(val_profiles)} | Test: {len(test_profiles)}")
    print("Gender distribution:", Counter(p["gender"] for p in profiles))
    print("Decade distribution:", Counter((p["birth_year"] // 10) * 10 for p in profiles))
    print("Background distribution:", Counter(p["cultural_background"] for p in profiles))

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/val_profiles.json", "w") as f:
        json.dump(val_profiles, f, indent=2)
    with open("data/processed/test_profiles.json", "w") as f:
        json.dump(test_profiles, f, indent=2)

    print("\nSaved val_profiles.json and test_profiles.json")
