from openai import OpenAI
import json
import random

client = OpenAI()

PROFILE_GEN_PROMPT = """Generate {n} fictional patient profiles for a music therapy dementia study.
Each profile must have these exact fields:
- name (realistic full name matching the cultural background)
- gender ("Male" or "Female")
- birth_year (between {year_min} and {year_max})
- hometown (a specific US city and state)
- cultural_background (one of: {backgrounds})
- life_events (list of 3-5 events, each with "year" and "event" keys)

CRITICAL REQUIREMENTS — read carefully:

Diversity of occupation and life path (do NOT default to stereotypes):
- Vary occupations across: factory worker, nurse, secretary, truck driver, seamstress,
  janitor, farmer, cook, mechanic, teacher, postal worker, salesperson, musician,
  domestic worker, carpenter, bus driver, store clerk, plumber, soldier, fisherman,
  hair stylist, laundry worker, coal miner, hotel worker — not just doctors/lawyers/engineers
- Vary socioeconomic background: working class, lower-middle class, and middle class.
  Not everyone went to college. Many worked from a young age or supported family early.
- Vary family structure: single parents, large families, people who never married,
  people who divorced, people who lost children or spouses early
- Vary life paths: some migrated for work, some stayed in their hometown their whole life,
  some served in the military, some were involved in community or religious life,
  some faced hardship (job loss, illness, discrimination, poverty)

Realism by era:
- Birth years {year_min}–{year_max} means formative years were roughly 1950–1990
- Life events must be historically plausible for the era, region, and background
- Reflect real historical context: civil rights movement, Vietnam War, deindustrialization,
  immigration waves, urban migration, suburbanization
- Do NOT give every profile a college education — that was not the norm for this generation,
  especially for working-class and minority communities

Avoid stereotypes:
- Do NOT assign roles based on ethnicity (e.g., no "opened a Chinese restaurant",
  no "became a doctor" for South Asian profiles by default)
- Names should feel natural and specific to the person's background and era,
  not generic or exaggerated
- Profiles should feel like real, specific individuals — not archetypes

Life events should span ages 15–35 and include a mix of: work, family, community,
migration, hardship, and personal milestones.

Respond with a JSON array of {n} profiles only, no other text.
"""

BACKGROUNDS = [
    "African American", "Mexican American", "Puerto Rican American",
    "Cuban American", "Dominican American", "Haitian American",
    "Korean American", "Chinese American", "Filipino American",
    "Vietnamese American", "Japanese American", "Cambodian American",
    "Italian American", "Polish American", "Greek American",
    "Irish American", "Portuguese American", "Hungarian American",
    "Indian American", "Pakistani American",
    "White American (Appalachian)", "White American (Midwestern)",
    "White American (Southern)", "Native American", "Arab American"
]

REGIONS = [
    "Detroit, Michigan", "Los Angeles, California", "Chicago, Illinois",
    "New York City, New York", "Houston, Texas", "Memphis, Tennessee",
    "San Francisco, California", "Atlanta, Georgia", "Miami, Florida",
    "Philadelphia, Pennsylvania", "Boston, Massachusetts", "New Orleans, Louisiana",
    "Seattle, Washington", "Cleveland, Ohio", "Newark, New Jersey",
    "Minneapolis, Minnesota", "Denver, Colorado", "Dallas, Texas",
    "Birmingham, Alabama", "Pittsburgh, Pennsylvania", "St. Louis, Missouri",
    "San Antonio, Texas", "El Paso, Texas", "Fresno, California",
    "Louisville, Kentucky", "Buffalo, New York", "Honolulu, Hawaii"
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