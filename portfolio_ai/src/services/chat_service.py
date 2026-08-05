from src.services.rag_service import answer_question


def ask_portfolio(question: str) -> str:
    """Validate a public chat request and delegate RAG work to the RAG service."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    return answer_question(question)
