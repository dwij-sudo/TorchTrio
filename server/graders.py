from typing import List
import re


def grade_classification(pred: str, expected: str) -> float:
    return 1.0 if pred.strip().lower() == expected.strip().lower() else 0.0


def grade_response(text: str, keywords: List[str]) -> float:
    lower = text.lower()
    coverage_hits = sum(1 for kw in keywords if kw.lower() in lower)
    coverage = coverage_hits / max(1, len(keywords))
    tone = 1.0 if re.search(r"\b(sorry|apologize|please|thanks)\b", lower) else 0.0
    return 0.6 * coverage + 0.4 * tone


def grade_escalation(level: str, expected: str) -> float:
    return 1.0 if level == expected else 0.0


def grade_priority(level: str, expected: str) -> float:
    return 1.0 if level == expected else 0.0
