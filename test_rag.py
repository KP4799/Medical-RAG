from services.rag_pipeline import RAGPipeline

rag = RAGPipeline()

while True:
    question = input("\nQuestion: ")

    result = rag.answer_question(question)

    print("\nANSWER\n")
    print(result["answer"])

    print("\nSOURCES\n")

    for source in result["sources"]:

        print(
            f"{source['source']} "
            f"(Page {source['page']})"
        )