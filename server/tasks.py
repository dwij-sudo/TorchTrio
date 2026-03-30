import json
import random
from pathlib import Path
from typing import List, Dict
from .models import TaskName

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.json"


def _load_all() -> List[Dict]:
    with _DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_tickets(task: TaskName, rng: random.Random) -> List[Dict]:
    tickets = _load_all()
    if task == "easy":
        subset = [t for t in tickets if t["category"] in {"billing", "technical", "general"}][:4]
    elif task == "medium":
        subset = tickets[:6]
    else:
        subset = tickets
    rng.shuffle(subset)
    return subset
