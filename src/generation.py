from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()


def run_variant_a(profile, baseline_prompt):
    """Variant A: Baseline LLM with no retrieval."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a music therapist. Respond with valid JSON only."},
            {"role": "user", "content": baseline_prompt.format(
                profile=json.dumps(profile, indent=2)
            )},
        ],
        temperature=0.4,
    )
    text = response.choices[0].message.content
    text = text.strip().strip("```json").strip("```").strip()
    return json.loads(text)


def run_variant_b(profile, faiss_index, bm25, df, generation_prompt, method="dense", k=20):
    """Variant B: RAG with birth-year-only query. Method and k are tunable."""
    from src.retrieval import retrieve
    bump_start = profile["birth_year"] + 15
    bump_end = profile["birth_year"] + 25
    query = f"popular songs {bump_start}-{bump_end}"
    retrieved = retrieve(query, faiss_index, df, k=k, method=method, bm25_index=bm25)
    result = generate_playlist(profile, retrieved, generation_prompt)
    return result, retrieved


def run_variant_c(profile, faiss_index, bm25, df, generation_prompt, method="dense", k_per_query=10, total_k=20):
    """Variant C: RAG with full biographical context. Method and k are tunable."""
    from src.profiling import profile_to_context
    retrieved, queries = profile_to_context(
        profile, faiss_index, bm25, df,
        method=method, k_per_query=k_per_query, total_k=total_k
    )
    result = generate_playlist(profile, retrieved, generation_prompt)
    return result, retrieved, queries


def generate_playlist(profile, retrieved_songs_df, prompt_template, retries=3):
    songs_text = ""
    for _, row in retrieved_songs_df.iterrows():
        songs_text += f"- {row['text_chunk']}\n"

    messages = [
        {"role": "system", "content": "You are a music therapist. Respond with valid JSON only."},
        {"role": "user", "content": prompt_template.format(
            profile=json.dumps(profile, indent=2),
            retrieved_songs=songs_text,
        )},
    ]

    for attempt in range(retries):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.4,
        )
        text = response.choices[0].message.content
        text = text.strip().strip("```json").strip("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt < retries - 1:
                print(f"  JSON parse failed (attempt {attempt+1}/{retries}), retrying...")
            else:
                raise
