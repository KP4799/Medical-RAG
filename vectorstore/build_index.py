import numpy as np
import faiss

EMBEDDINGS_FILE = "data/processed/embeddings.npy"
INDEX_FILE = "data/processed/faiss_index.bin"

def build_faiss_index():
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Loaded embeddings: {embeddings.shape}")
    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"Vectors indexed: {index.ntotal}")

    faiss.write_index(index,INDEX_FILE)

    print(f"Indices saved to {INDEX_FILE}")

if __name__ == "__main__":
    build_faiss_index()