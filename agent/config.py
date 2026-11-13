import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKAB_DATA = Path(os.environ.get("SKAB_DATA", ROOT.parent / "SKAB" / "data"))
DB_PATH = Path(os.environ.get("SKAB_AGENT_DB", ROOT / "skab_agent.db"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
EMBED_MODEL = "nomic-embed-text-v2-moe"
OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "0").lower() in ("1", "true", "yes")
ARTIFACTS = Path(os.environ.get("SKAB_AGENT_ARTIFACTS", ROOT / "artifacts"))
