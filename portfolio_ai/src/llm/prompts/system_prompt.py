import sys
from pathlib import Path

# Ensure portfolio_ai root directory is in sys.path
PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config.settings import RAW_DATA_DIR

SYSTEM_CONTEXT_FILE = RAW_DATA_DIR / "system_context.md"

def get_system_prompt() -> str:
    base_instructions = (
        "You are Vignesh's portfolio assistant.\n"
        "Answer questions based on the supplied portfolio context.\n"
        "If the context is insufficient, politely state that the information isn't available in the portfolio."
    )
    if SYSTEM_CONTEXT_FILE.exists():
        file_content = SYSTEM_CONTEXT_FILE.read_text(encoding="utf-8").strip()
        if file_content:
            return file_content

    return base_instructions

SYSTEM_PROMPT = get_system_prompt()
