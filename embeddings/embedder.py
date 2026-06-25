import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/processed/chunks.json"
EMBEDDINGS_FILE = "data/processed/embeddings.npy"
METADATA_FILE = "data/processed/metadata.json"

def load_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    
def generate_embeddings(chunks):
    print("Loading embedding model")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]

    print(f"\nGenerating embeddings for {len(chunks)} chunks")

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def save_embeddings(embeddings):
    np.save(EMBEDDINGS_FILE, embeddings)
    print("\nSaved metadata file")
    print(EMBEDDINGS_FILE)

def save_metadata(chunks):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    
    print("\nSaved metadata file")
    print(METADATA_FILE)

if __name__ == "__main__":
    chunks = load_chunks()
    embeddings = generate_embeddings(chunks)

    save_embeddings(embeddings)
    save_metadata(chunks)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print("\nEmbeddings Generated")
    print("\nMetadata Saved")
    