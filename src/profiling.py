from openai import OpenAI
import json
import pandas as pd

client = OpenAI()

# write the prompt fed to openai api to find songs based on music base 
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

# function that creates the queries given a profile
def generate_queries(profile):
    # defining bump period
    bump_start = profile["birth_year"] + 15
    bump_end = profile["birth_year"] + 25
    profile_with_bump = {**profile, "reminiscence_bump": f"{bump_start}-{bump_end}"}

    # collect output
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
    # clean queries
    text = response.choices[0].message.content
    text = text.strip().strip("```json").strip("```").strip()
    queries = json.loads(text)
    return queries


# function that is the full pipeline of profile -> queries -> retrieved songs 
def profile_to_context(profile, index, df, k_per_query=10, total_k=20):
    from src.retrieval import retrieve

    # generate queries
    queries = generate_queries(profile)
    print(f"Generated {len(queries)} queries:")
    for q in queries:
        print(f"  - {q}")

    # retrieve for each query
    all_results = []
    for query in queries:
        results = retrieve(query, index, df, k=k_per_query)
        results["source_query"] = query
        all_results.append(results)

    #combine and deduplicate and keep the highest similarity
    combined = pd.concat(all_results)
    combined = (
        combined
        .sort_values("similarity_score", ascending=False)
        .drop_duplicates(subset=["song", "artist"], keep="first")
        .head(total_k)
    )

    return combined, queries
