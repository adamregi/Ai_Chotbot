from src.llm.factory import get_llm
from src.llm.prompts.rag_prompt import build_rag_prompt
from src.llm.prompts.system_prompt import SYSTEM_PROMPT
from src.vectorstore.chroma_repository import ChromaRepository

repository = ChromaRepository()
llm = get_llm()


def retrieve_context(question: str) -> str:
    results = repository.search(question)
    return "\n\n".join(str(result["text"]) for result in results)


def answer_question(question: str) -> str:
    context = retrieve_context(question)
    if not context:
        return "I couldn't find any relevant information."
    return llm.generate(SYSTEM_PROMPT, build_rag_prompt(context, question))
