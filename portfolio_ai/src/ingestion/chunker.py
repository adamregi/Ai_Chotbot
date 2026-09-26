from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter


class FixedWindowTextSplitter(TextSplitter):
    """Text splitter that creates fixed-size windows with exact character overlap."""

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        step = self._chunk_size - self._chunk_overlap
        return [
            text[start : start + self._chunk_size]
            for start in range(0, len(text), step)
        ]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size.")
    if not text:
        return []

    splitter = FixedWindowTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
