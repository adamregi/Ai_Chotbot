def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size.")
    if not text:
        return []

    return [
        text[start : start + chunk_size]
        for start in range(0, len(text), chunk_size - overlap)
    ]
