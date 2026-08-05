from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vectorstore.chroma_repository import ChromaRepository


if __name__ == "__main__":
    question = "Tell me about Vignesh's projects."
    repo = ChromaRepository()
    count = repo.collection.count()
    print(f"Total documents in collection: {count}")
    results = repo.search(question)
    for index, result in enumerate(results, start=1):
        print(f"Result {index}\n{'-' * 50}")
        print(f"Metadata: {result.get('metadata')}")
        print(f"Distance: {result.get('distance')}")
        print(f"Text:\n{result.get('text')}\n")
