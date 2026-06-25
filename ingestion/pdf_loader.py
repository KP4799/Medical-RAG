import os
import io
import pytesseract
from PIL import Image
import json
import fitz

DATA_DIR = "data"
OUTPUT_FILE = "data/processed/extracted_documents.json"

def extract_page_with_ocr(page):
    # Render page as image
    pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(image)
    return text

def extract_documents():
    documents = []

    # Looping through topic folders
    for topic in os.listdir(DATA_DIR):
        topic_path = os.path.join(DATA_DIR, topic)

        # Skip files and processed folder
        if not os.path.isdir(topic_path) or topic == "processed":
            continue
        
        print(f"Processing topic: {topic}")

        # Loop through pdfs for a particular topic
        for filename in os.listdir(topic_path):
            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(topic_path, filename)
            print(f"Loading: {filename}")

            try:
                pdf = fitz.open(pdf_path)
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    text = page.get_text()

                    method = "TEXT"
                    if len(text.strip()) < 50:
                        print(f"OCR Fallback: Page {page_num + 1}")
                        method = "OCR"
                        text = extract_page_with_ocr(page)

                    document = {
                        "source": filename,
                        "topic": topic,
                        "page": page_num + 1,
                        "char_count": len(text),
                        "method": method,
                        "text": text.strip()
                    }

                    documents.append(document)
                pdf.close()

            except Exception as e:
                print(f"Error processing {filename}: {e}")
            
    return documents

def save_documents(documents):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(documents)} pages")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    docs = extract_documents()
    save_documents(docs)

    print(f"Sample record:")

    if docs:
        sample = docs[0]
        print(f"Source: {sample['source']}")
        print(f"Topic: {sample['topic']}")
        print(f"Page: {sample['page']}")
        print(f"Characters: {sample['char_count']}")

        print("\nText Preview:\n")
        print(sample["text"][:500])