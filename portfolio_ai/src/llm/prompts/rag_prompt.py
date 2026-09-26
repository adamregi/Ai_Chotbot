from langchain_core.prompts import ChatPromptTemplate

RAG_USER_TEMPLATE = """PORTFOLIO CONTEXT
========================
{context}

QUESTION
========================
{question}

ANSWER
========================
"""


def get_rag_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    """Return a LangChain ChatPromptTemplate configured with system and RAG user messages."""
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", RAG_USER_TEMPLATE),
    ])


def build_rag_prompt(context: str, question: str) -> str:
    """Format context and question into the RAG prompt string."""
    return RAG_USER_TEMPLATE.format(context=context, question=question)
