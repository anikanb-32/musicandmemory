from openai import OpenAI
import json

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