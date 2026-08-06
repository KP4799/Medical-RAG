import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "data/processed/faiss_index.bin"
CHUNK_FILE = "data/processed/chunks.json"

TOP_K = 15

class Retriever:
    _model = None

    def __init__(self):
        if Retriever._model is None:
            print("Loading Embedding Model")
            Retriever._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.model = Retriever._model
        
        print("Loading FAISS Index")
        self.index = faiss.read_index(INDEX_FILE)

        with open(CHUNK_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
    def search(self, query, top_k=TOP_K):
        query_embedding = self.model.encode([query],convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding,top_k)

        seen = set()
        results = []

        for idx in indices[0]:
            chunk = self.chunks[idx]
            key = (chunk["source"],chunk["page"])

            if key not in seen:
                results.append(chunk)
                seen.add(key)

            if len(results) >= top_k:
                break

        return results
    
if __name__ == "__main__":
    retriever = Retriever()

    while True:
        query = input("\nQuestion: ")
        results = retriever.search(query)

        print("\n", "=" * 80)

        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}:")

            print(f"Source: {result['source']}")
            print(f"Page: {result['page']}")
            print(f"Topic: {result['topic']}")
            print("\nChunk:")
            print(result["text"])

            print("\n" + "-" * 80)

    