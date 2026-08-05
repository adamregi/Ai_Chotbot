from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.ingest import ingest


if __name__ == "__main__":
    print(f"Portfolio ingested successfully ({ingest()} chunks).")

