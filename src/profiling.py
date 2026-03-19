from openai import OpenAI
import json
import pandas as pd

client = OpenAI()

QUERY_GENERATION_PROMPT = """You are a music therapy specialist. Given a patient profile, generate retrieval queries to find personally meaningful songs from their life.

RULES:
1. Focus on the REMINISCENCE BUMP (ages 15–25) — this is when music memories are strongest
2. Consider their geographic region (local radio stations played regional hits)
3. Map life events to time periods (wedding songs, graduation year hits, etc.)
4. Consider cultural background (what genres/artists were popular in their community)
5. Generate 5–8 specific queries

PATIENT PROFILE:
{profile}

Respond with a JSON array of query strings only. Example:
["popular Motown hits in Detroit 1963-1968", "top R&B songs 1965", ...]
"""


def generate_queries(profile):
    """Generate retrieval queries from a patient profile."""
    bump_start = profile["birth_year"] + 15
    bump_end = profile["birth_year"] + 25
    profile_with_bump = {**profile, "reminiscence_bump": f"{bump_start}-{bump_end}"}

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You generate music retrieval queries. Respond with JSON only."},
            {"role": "user", "content": QUERY_GENERATION_PROMPT.format(
                profile=json.dumps(profile_with_bump, indent=2)
            )},
        ],
        temperature=0.3,
    )

    text = response.choices[0].message.content
    text = text.strip().strip("```json").strip("```").strip()
    queries = json.loads(text)
    return queries


def profile_to_context(profile, index, df, k_per_query=10, total_k=20):
    """Full pipeline: profile → queries → retrieved songs."""
    from src.retrieval import retrieve

    # Step 1: Generate queries
    queries = generate_queries(profile)
    print(f"Generated {len(queries)} queries:")
    for q in queries:
        print(f"  - {q}")

    # Step 2: Retrieve for each query
    all_results = []
    for query in queries:
        results = retrieve(query, index, df, k=k_per_query)
        results["source_query"] = query
        all_results.append(results)

    # Step 3: Combine and deduplicate, keeping highest similarity
    combined = pd.concat(all_results)
    combined = (
        combined
        .sort_values("similarity_score", ascending=False)
        .drop_duplicates(subset=["song", "artist"], keep="first")
        .head(total_k)
    )

    return combined, queries
