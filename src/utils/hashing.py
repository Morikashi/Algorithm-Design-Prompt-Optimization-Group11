import hashlib


def hash_text(text: str) -> str:
    """Stable hash for deduping prompts."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
