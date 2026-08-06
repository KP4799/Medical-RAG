import re
import json

INPUT_FILE = "data/processed/extracted_documents.json"
OUTPUT_FILE = "data/processed/cleaned_documents.json"

def clean_text(text):
    # Fix words with broken lines (eg. indi-vidual -> individual)
    text = re.sub(r'-\s*\n\s*','',text)

    # Replace remaining new lines with spaces
    text = text.replace("\n"," ")

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def clean_document(document):
    return {
        **document,
        "text": clean_text(document["text"])
    }

def clean_documents():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    cleaned_documents = [clean_document(document) for document in documents]
    return cleaned_documents

def save_documents(documents):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(documents)} cleaned documents")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    cleaned_docs = clean_documents()
    save_documents(cleaned_docs)

    print("\nSample of Cleaned Text:\n")
    print(cleaned_docs[0]["text"][:1000])