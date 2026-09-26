from langchain_core.output_parsers import StrOutputParser
from src.llm.factory import get_chat_model, get_llm
from src.llm.prompts.rag_prompt import get_rag_prompt_template, build_rag_prompt
from src.llm.prompts.system_prompt import SYSTEM_PROMPT
from src.vectorstore.chroma_repository import ChromaRepository

repository = ChromaRepository()
prompt_template = get_rag_prompt_template(SYSTEM_PROMPT)


def format_docs(docs) -> str:
    """Format retrieved LangChain documents into readable context chunks."""
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source") if hasattr(doc, "metadata") and doc.metadata else None
        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
        if source:
            formatted_chunks.append(f"--- Document: {source} ---\n{text}")
        else:
            formatted_chunks.append(text)
    return "\n\n".join(formatted_chunks)


def retrieve_context(question: str) -> str:
    doc_count = repository.collection.count() if repository.collection else 0
    print(f"[RAG] Chroma documents in index: {doc_count}")
    if doc_count == 0:
        print("[WARNING] Chroma collection 'portfolio' has 0 documents! Run 'python scripts/rebuild_db.py' locally.")
        return ""

    try:
        retriever = repository.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question)
        if docs:
            return format_docs(docs)
    except Exception as exc:
        print(f"[WARNING] Retriever invoke failed, falling back to repository search: {exc}")

    results = repository.search(question)
    formatted_chunks = []
    for res in results:
        source = res.get("metadata", {}).get("source")
        text = res.get("text", "")
        if source:
            formatted_chunks.append(f"--- Document: {source} ---\n{text}")
        else:
            formatted_chunks.append(text)

    return "\n\n".join(formatted_chunks)


def answer_question(question: str) -> str:
    context = retrieve_context(question)
    if not context:
        return "I couldn't find any relevant information."

    try:
        chat_model = get_chat_model()
        chain = prompt_template | chat_model | StrOutputParser()
        return chain.invoke({"context": context, "question": question})
    except Exception as exc:
        # Fallback to direct client generation if LCEL chat model is unavailable
        print(f"[RAG] LangChain chat model generation failed: {exc}, using client generate")
        return get_llm().generate(SYSTEM_PROMPT, build_rag_prompt(context, question))
