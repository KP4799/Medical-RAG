from retrieval.retriever import Retriever
from llm.generator import Generator
from llm.question_rewriter import Rewriter

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.rewriter = Rewriter()
        print("\nPipeline Loaded Successfully")

    def answer_question(self, question, chat_history=None):
        if not chat_history:
            rewritten_question = question
        else:
            rewritten_question = self.rewriter.rewrite(question, chat_history)

        if rewritten_question.startswith("LLM Error"):
            rewritten_question = question

        search_question = rewritten_question or question




        print("\nOriginal Question:", question)
        print("Rewritten Question:", rewritten_question)
        print("Search Question:", search_question)




        retrieved_chunks = self.retriever.search(search_question,top_k=5)



        print("\nRetrieved Chunks:")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"\nChunk {i+1}")
            print("Source:", chunk["source"])
            print("Topic:", chunk["topic"])
            print(chunk["text"][:250])





        retrieved_context = "\n\n".join(chunk["text"]for chunk in retrieved_chunks)
        answer = self.generator.generate(search_question,retrieved_context)

        llm_available = True
        if answer is None:
            llm_available = False
            answer = ("⚠️ The AI model is temporarily unavailable.\n\n"
            "Relevant evidence from the knowledge base is shown below.")

        NO_INFO_MESSAGE = (
            "I could not find sufficient information "
            "in the provided documents."
        )

        if NO_INFO_MESSAGE in answer:
            return {
                "answer": answer,
                "llm_available": llm_available,
                "rewritten_question": rewritten_question,
                "sources": sources,
                "chunks": retrieved_chunks
            }
        
        sources = []
        for chunk in retrieved_chunks:
            source = {
                "source": chunk["source"],
                "page": chunk["page"],
                "topic": chunk["topic"]
            }

            if source not in sources:
                sources.append(source)

        return {
            "answer": answer,
            "llm_available": llm_available,
            "rewritten_question": rewritten_question,
            "sources": sources,
            "chunks": retrieved_chunks
        }
    