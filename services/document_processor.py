import json
from ingestion.pdf_loader import extract_pdf
from ingestion.cleaner import clean_document
from ingestion.chunker import create_chunks_for_document
from embeddings.embedder import Embedder
from vectorstore.build_index import load_index, save_index, add_embeddings

CHUNKS_FILE = "data/processed/chunks.json"

class DocumentProcessor:
    def __init__(self):
        self.embedder = Embedder()

    def process_pdf(self, pdf_path, topic):
        documents = extract_pdf(pdf_path, topic)
        cleaned_documents = [clean_document(doc) for doc in documents]

        with open(CHUNKS_FILE,"r") as f:
            existing_chunks = json.load(f)

        next_chunk_id = len(existing_chunks)
        current_chunk_id = next_chunk_id
        all_chunks = []

        for document in cleaned_documents:
            new_chunks = create_chunks_for_document(document, current_chunk_id)
            all_chunks.extend(new_chunks)
            current_chunk_id += len(new_chunks)

        embeddings = self.embedder.embed_chunks(all_chunks)

        index = load_index()
        add_embeddings(index, embeddings)
        save_index(index)
        print(f"Index now contains {index.ntotal} vectors")

        existing_chunks.extend(all_chunks)

        with open(CHUNKS_FILE, "w") as f:
            json.dump(existing_chunks,f,indent=4,ensure_ascii=False)

        return {
            "pages":len(cleaned_documents),
            "chunks":len(all_chunks)
        }