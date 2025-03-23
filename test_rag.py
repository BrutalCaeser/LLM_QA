from src.rag.query_engine import BookingQA


def test_rag():
    qa_system = BookingQA()

    questions = [
        "What is the average lead time for Resort Hotel bookings?",
        "Show me stays longer than 3 nights in India ( IND )",
        "Show 2017 cancellations from Germany ( DEU) with ADR greater than 150",
        "Are there any cancellations from FRA in 2017"
        ]


    for question in questions:
        response = qa_system.ask(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {response['result']}")
        print("Sources:")
        for doc in response['source_documents'][:2]:
            print(f"- {doc.page_content[:100]}...")


if __name__ == "__main__":
    test_rag()