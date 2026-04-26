from openai import OpenAI
import json
import pandas as pd

client = OpenAI(timeout=60.0)

def generate_playlist(profile, retrieved_songs_df, prompt_template, retries=3):
    songs_text = ""
    for _, row in retrieved_songs_df.iterrows():
        songs_text += f"- {row['text_chunk']}\n"

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a music therapist. Respond with valid JSON only."},
                    {"role": "user", "content": prompt_template.format(
                        profile=json.dumps(profile, indent=2),
                        retrieved_songs=songs_text,
                    )},
                ],
                temperature=0.4,
            )
            text = response.choices[0].message.content
            text = text.strip().strip("```json").strip("```").strip()
            return json.loads(text)
        except Exception as e:
            if attempt < retries - 1:
                print(f"    generate_playlist attempt {attempt+1} failed: {e}, retrying...")
            else:
                raise


def run_variant_a(profile, prompt_template):
    """Baseline — no retrieval."""
    return generate_playlist(profile, pd.DataFrame(), prompt_template)


def run_variant_b(profile, faiss_index, bm25, df, prompt_template, method="dense", k=20):
    """RAG with a single birth-year temporal query."""
    from src.retrieval import retrieve
    bump_start = profile["birth_year"] + 15
    bump_end   = profile["birth_year"] + 25
    query      = f"popular songs {bump_start}-{bump_end}"
    retrieved  = retrieve(query, faiss_index, df, k=k, method=method, bm25_index=bm25)
    result     = generate_playlist(profile, retrieved, prompt_template)
    return result, retrieved


def run_variant_c(profile, faiss_index, bm25, df, prompt_template,
                  method="dense", k_per_query=6, total_k=30):
    """RAG with multiple profile-derived queries."""
    from src.profiling import profile_to_context
    retrieved, queries = profile_to_context(
        profile, faiss_index, bm25, df,
        method=method, k_per_query=k_per_query, total_k=total_k,
    )
    result = generate_playlist(profile, retrieved, prompt_template)
    return result, retrieved, queries