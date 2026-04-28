"""
Flask backend for the Music & Memory playlist generator interface.
Run: python3 app.py
Then open interface.html in your browser.
"""
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json, os

from src.retrieval import load_retrieval_system
from src.generation import run_variant_c
from configs.prompts import GENERATION_PROMPT

app = Flask(__name__)
CORS(app)

print("Loading retrieval system...")
faiss_index, df, bm25 = load_retrieval_system(
    "data/index/songs.index",
    "data/processed/knowledge_base.csv",
    "data/index/bm25.pkl"
)

with open("outputs/best_configs.json") as f:
    best_configs = json.load(f)

cfg = best_configs["variant_c"]
print(f"Using Variant C: method={cfg['method']}, k={cfg['k']}")
print("Ready. Open interface.html in your browser.\n")


@app.route("/")
def index():
    return send_file("index.html")

@app.route("/styles.css")
def styles():
    return send_file("styles.css", mimetype="text/css")

@app.route("/script.js")
def script():
    return send_file("script.js", mimetype="application/javascript")


@app.route("/generate", methods=["POST"])
def generate():
    profile = request.get_json()
    if not profile:
        return jsonify({"error": "No profile provided"}), 400

    required = ["name", "gender", "birth_year", "hometown", "cultural_background", "life_events"]
    missing = [f for f in required if f not in profile]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        result, _, _ = run_variant_c(
            profile, faiss_index, bm25, df, GENERATION_PROMPT,
            method=cfg["method"],
            k_per_query=max(cfg["k"] // 5, 5),
            total_k=cfg["k"]
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5050, debug=False)
