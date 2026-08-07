import json
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except:
    nltk.download("tokenizers/punkt_tab")

INPUT_FILE = "data/processed/cleaned_documents.json"
OUTPUT_FILE = "data/processed/chunks.json"

CHUNK_SIZE = 800
MIN_CHUNK_LENGTH = 100
OVERLAP_SENTENCES = 2

def chunk_text(text):
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length <= CHUNK_SIZE:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            overlap = current_chunk[-OVERLAP_SENTENCES:]
            current_chunk = overlap + [sentence]
            current_length = sum(len(s) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def create_chunks_for_document(document, start_chunk_id=0):
    chunked_docs = []
    chunk_id = start_chunk_id

    chunks = chunk_text(document["text"])

    for chunk in chunks:
        if len(chunk.strip()) < MIN_CHUNK_LENGTH:
            continue

        chunked_docs.append({
            "chunk_id": chunk_id,
            "source": document["source"],
            "topic": document["topic"],
            "page": document["page"],
            "text": chunk
        })
        chunk_id += 1

    return chunked_docs

def create_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    chunked_docs = []
    chunk_id = 0

    for document in documents:
        new_chunks = create_chunks_for_document(document,chunk_id)
        chunked_docs.extend(new_chunks)
        chunk_id += len(new_chunks)

    return chunked_docs

def save_chunks(chunks):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    chunks = create_chunks()
    save_chunks(chunks)

    print("\nSample chunk:\n")
    print(chunks[0]["text"][:1000])
