from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vectorstore.chroma_repository import ChromaRepository


if __name__ == "__main__":
    question = "Tell me about Vignesh's projects."
    results = ChromaRepository().search(question)
    for index, result in enumerate(results, start=1):
        print(f"Result {index}\n{'-' * 50}\nDistance: {result['distance']}\n{result['text']}\n")
