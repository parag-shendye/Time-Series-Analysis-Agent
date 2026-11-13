import ollama

from .config import OLLAMA_HOST, OLLAMA_MODEL


def client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def installed_models() -> list[dict]:
    """Models pulled into the container (GET /api/tags), with capabilities."""
    return [m.model_dump() for m in client().list().models]


def running_models() -> list[str]:
    """Models currently loaded in memory (GET /api/ps)."""
    return [m.model for m in client().ps().models]


def check() -> None:
    """Fail early if Ollama is unreachable or the configured model isn't pulled."""
    names = [m["model"] for m in installed_models()]
    if OLLAMA_MODEL not in names:
        raise RuntimeError(f"{OLLAMA_MODEL} not pulled at {OLLAMA_HOST}; available: {names}")


if __name__ == "__main__":
    print("host:", OLLAMA_HOST)
    print("configured model:", OLLAMA_MODEL)
    print("installed:", [m["model"] for m in installed_models()])
    print("running:", running_models())
    check()
