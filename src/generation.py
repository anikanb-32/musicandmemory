from openai import OpenAI
import json

client = OpenAI()

def generate_playlist(profile, retrieved_songs_df, prompt_template):
    songs_text = ""
    for _, row in retrieved_songs_df.iterrows():
        songs_text += f"- {row['text_chunk']}\n"

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