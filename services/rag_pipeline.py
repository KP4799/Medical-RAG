from retrieval.retriever import Retriever
from llm.generator import Generator

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        print("\nPipeline Loaded Successfully")

    def answer_question(self, question):
        retrieved_chunks = self.retriever.search(question,top_k=15)
        context_chunks = retrieved_chunks[:3]
        context = "\n\n".join(chunk["text"]for chunk in context_chunks)
        answer = self.generator.generate(question,context)

        NO_INFO_MESSAGE = (
            "I could not find sufficient information "
            "in the provided documents."
        )

        if NO_INFO_MESSAGE in answer:
            return {
                "answer": answer,
                "sources": [],
                "chunks": []
            }
        
        sources = []
        for chunk in context_chunks:
            source = {
                "source": chunk["source"],
                "page": chunk["page"],
                "topic": chunk["topic"]
            }

            if source not in sources:
                sources.append(source)

        return {
            "answer": answer,
            "sources": sources,
            "chunks": context_chunks
        }
    