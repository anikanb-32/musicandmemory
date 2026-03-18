import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI()

def load_retrieval_system(index_path, kb_path):
    """Load the FAISS index and knowledge base."""
    index = faiss.read_index(index_path)
    df = pd.read_csv(kb_path)
    return index, df

def retrieve(query, index, df, k=20):
    """Retrieve top-k songs for a query string."""
    # Embed the query
    response = client.embeddings.create(
        input=[query], model="text-embedding-3-small"
    )
    query_vec = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(query_vec)

    # Search
    scores, indices = index.search(query_vec, k)

    # Return results
    results = df.iloc[indices[0]].copy()
    results["similarity_score"] = scores[0]
    return results
