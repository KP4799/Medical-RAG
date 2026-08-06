import numpy as np
import faiss

EMBEDDINGS_FILE = "data/processed/embeddings.npy"
INDEX_FILE = "data/processed/faiss_index.bin"

def load_embeddings():
    return np.load(EMBEDDINGS_FILE)

def load_index():
    return faiss.read_index(INDEX_FILE)

def save_index(index):
    faiss.write_index(index, INDEX_FILE)
    print(f"Index saved to {INDEX_FILE}")

def add_embeddings(index, embeddings):
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def build_index(embeddings):
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def build_faiss_index():
    embeddings = load_embeddings()
    print(f"Loaded embeddings: {embeddings.shape}")

    index = build_index(embeddings)
    print(f"Vectors indexed: {index.ntotal}")

    save_index(index)
    print(f"Indices saved to {INDEX_FILE}")

if __name__ == "__main__":
    build_faiss_index()