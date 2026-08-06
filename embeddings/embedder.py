import json
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/processed/chunks.json"
EMBEDDINGS_FILE = "data/processed/embeddings.npy"
METADATA_FILE = "data/processed/metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    _model = None

    def __init__(self):
        if Embedder._model is None:
            print("Loading embedding model")
            Embedder._model = SentenceTransformer(MODEL_NAME)
            
        self.model = Embedder._model

    def load_chunks(self):
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
        
    def embed_chunks(self,chunks):
        texts = [chunk["text"] for chunk in chunks]
        print(f"\nGenerating embeddings for {len(chunks)} chunks")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        return embeddings

    def generate_embeddings(self):
        chunks = self.load_chunks()
        embeddings = self.embed_chunks(chunks)
        return embeddings, chunks

    def save_embeddings(self,embeddings):
        np.save(EMBEDDINGS_FILE, embeddings)
        print("\nSaved embeddings file")
        print(EMBEDDINGS_FILE)

    def save_metadata(self,chunks):
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        
        print("\nSaved metadata file")
        print(METADATA_FILE)

if __name__ == "__main__":
    embedder = Embedder()
    embeddings, chunks = embedder.generate_embeddings()

    embedder.save_embeddings(embeddings)
    # embedder.save_metadata(chunks)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print("\nEmbeddings Generated")
    # print("\nMetadata Saved")
    