from pathlib import Path
import shutil
from services.document_processor import DocumentProcessor

class UploadService:
    def __init__(self):
        self.data_dir = Path("data")

    def upload_pdf(self, uploaded_file, topic):
        if uploaded_file.type != "application/pdf":
            raise ValueError("Only PDF files are supported.")
        
        topic = topic.strip().lower()
        topic_dir = self.data_dir / topic
        topic_dir.mkdir(parents=True,exist_ok=True)
        pdf_path = topic_dir / uploaded_file.name

        if pdf_path.exists():
            raise FileExistsError(f"{uploaded_file.name} already exists.")

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)

        try:
            document_processor = DocumentProcessor()
            print("Document processor loaded now trying to process the pdf")
            stats = document_processor.process_pdf(pdf_path, topic)
            print("Returned stats:", stats)
            return stats

        except Exception:
            if pdf_path.exists():
                pdf_path.unlink()
            raise