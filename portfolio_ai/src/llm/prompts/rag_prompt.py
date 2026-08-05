def build_rag_prompt(context: str, question: str) -> str:
    return f"""PORTFOLIO CONTEXT
========================
{context}

QUESTION
========================
{question}

ANSWER
========================
"""
