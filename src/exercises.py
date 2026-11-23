"""Pemuat definisi latihan berbicara dan bernyanyi dari resources/exercises.json."""

from functools import lru_cache
import json
from typing import Any, Dict, List

from src.config import RESOURCES_DIR


@lru_cache(maxsize=1)
def load_exercises() -> Dict[str, List[Dict[str, Any]]]:
    """Muat daftar latihan berbicara dan bernyanyi dari file JSON sumber (di-cache dalam memori)."""
    exercises_path = RESOURCES_DIR / "exercises.json"
    with exercises_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {
        "speaking_exercises": data.get("speaking_exercises", []),
        "singing_exercises": data.get("singing_exercises", []),
    }
