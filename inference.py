import os
import sys
from typing import Any
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback for environments without openai installed
    OpenAI = None
from server.env import SupportOpsEnv

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
SCORE_EPSILON = 0.01


def emit_block(tag: str, **fields: Any) -> None:
    parts = []
    for key, value in fields.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, float):
            rendered = f"{value:.2f}"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    suffix = f" {' '.join(parts)}" if parts else ""
    print(f"[{tag}]{suffix}", flush=True)


def _compact_error(exc: Exception) -> str:
    raw = str(exc).strip() or exc.__class__.__name__
    return raw.replace("\r", " ").replace("\n", " ").replace(" ", "_")[:160]


def _to_open_score(raw_reward_score: float) -> float:
    # Convert reward range [-1, 1] to score range [0, 1], then keep it strictly inside bounds.
    normalized = (raw_reward_score + 1.0) / 2.0
    return min(1.0 - SCORE_EPSILON, max(SCORE_EPSILON, normalized))


def heuristic_classify(ticket_text: str) -> str:
    """Fallback classification when LLM is unavailable."""
    text_lower = ticket_text.lower()
    
    # Check for billing-related keywords
    billing_keywords = ["bill", "charge", "payment", "invoice", "refund", "credit card", "subscription", "price"]
    if any(keyword in text_lower for keyword in billing_keywords):
        return "billing"
    
    # Check for technical-related keywords
    technical_keywords = ["error", "bug", "crash", "fix", "technical", "api", "code", "exception", "broken", "not working"]
    if any(keyword in text_lower for keyword in technical_keywords):
        return "technical"
    
    # Default to general
    return "general"


def maybe_client():
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to create OpenAI client: {e}") from e


def llm_classify(client, ticket_text: str, sentiment: str) -> str:
    prompt = (
        "Classify this customer support ticket into one of: billing, technical, general. "
        f"Ticket: {ticket_text}. Sentiment: {sentiment}. "
        "Respond with only the category name."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        content = resp.choices[0].message.content.strip().lower()
        if content in ["billing", "technical", "general"]:
            return content
        else:
            return heuristic_classify(ticket_text)
    except Exception as e:
        print(f"LLM classification failed: {e}", file=sys.stderr, flush=True)
        return heuristic_classify(ticket_text)


def llm_escalate(client, ticket_text: str, sentiment: str, task: str) -> str:
    prompt = (
        "Should this customer support ticket be escalated? Levels: none, tier1, tier2, tier3. "
        f"Ticket: {ticket_text}. Sentiment: {sentiment}. Task: {task}. "
        "Respond with only the level name."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        content = resp.choices[0].message.content.strip().lower()
        if content in ["none", "tier1", "tier2", "tier3"]:
            return content
        else:
            return "none"
    except Exception as e:
        print(f"LLM escalation failed: {e}", file=sys.stderr, flush=True)
        return "none"


def llm_priority(client, ticket_text: str, sentiment: str, task: str) -> str:
    prompt = (
        "Set priority for this customer support ticket: low, medium, high, urgent. "
        f"Ticket: {ticket_text}. Sentiment: {sentiment}. Task: {task}. "
        "Respond with only the priority level."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        content = resp.choices[0].message.content.strip().lower()
        if content in ["low", "medium", "high", "urgent"]:
            return content
        else:
            return "medium"
    except Exception as e:
        print(f"LLM priority failed: {e}", file=sys.stderr, flush=True)
        return "medium"


def llm_response(client, ticket_text: str, sentiment: str, task: str) -> str:
    prefix = "Sorry" if sentiment in ("angry", "negative") else "Thanks"
    fallback = f"{prefix} for reaching out. We are investigating: {ticket_text[:120]} and will update you shortly."
    prompt = (
        "You are a concise, empathetic SaaS support agent. "
        f"Task: {task}. Sentiment: {sentiment}. Ticket: {ticket_text}. "
        "Respond briefly with apology if negative and include next steps."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
        )
        content = resp.choices[0].message.content
        return content if content else fallback
    except Exception as e:
        print(f"LLM request failed: {e}", file=sys.stderr, flush=True)
        return fallback


def deterministic_response(keywords: list[str], sentiment: str) -> str:
    tone = "Sorry" if sentiment in ("angry", "negative") else "Thanks"
    keyword_phrase = ", ".join(keywords)
    return (
        f"{tone} for reaching out. Please know we are investigating this now. "
        f"Next steps: {keyword_phrase}."
    )


def run_task(env: SupportOpsEnv, task: str, client) -> float:
    total_reward = 0.0
    step_count = 0
    tickets = 0
    started = False

    try:
        obs = env.reset(task=task)
        tickets = len(env.tickets)
        emit_block("START", task=task, tickets=tickets)
        started = True

        for _ in range(tickets):
            ticket = env.tickets[env.current_index]
            ticket_id = ticket["id"]

            # classification
            category = llm_classify(client, obs.ticket_text, obs.customer_sentiment)
            obs, r, done, _ = env.step({"action_type": "classify_ticket", "category": category})
            total_reward += r
            step_count += 1
            emit_block(
                "STEP",
                task=task,
                step=step_count,
                ticket=ticket_id,
                action="classify_ticket",
                reward=r,
            )

            # response
            if task in ("medium", "hard"):
                reply = llm_response(client, obs.ticket_text, obs.customer_sentiment, task)
                obs, r, done, _ = env.step({"action_type": "respond_ticket", "text": reply})
                total_reward += r
                step_count += 1
                emit_block(
                    "STEP",
                    task=task,
                    step=step_count,
                    ticket=ticket_id,
                    action="respond_ticket",
                    reward=r,
                )

            # escalation and priority for hard
            if task == "hard":
                esc = llm_escalate(client, obs.ticket_text, obs.customer_sentiment, task)

                if esc != "none":
                    obs, r, done, _ = env.step({"action_type": "escalate_ticket", "level": esc})
                    total_reward += r
                    step_count += 1
                    emit_block(
                        "STEP",
                        task=task,
                        step=step_count,
                        ticket=ticket_id,
                        action="escalate_ticket",
                        reward=r,
                    )

                pri = llm_priority(client, obs.ticket_text, obs.customer_sentiment, task)
                obs, r, done, _ = env.step({"action_type": "set_priority", "level": pri})
                total_reward += r
                step_count += 1
                emit_block(
                    "STEP",
                    task=task,
                    step=step_count,
                    ticket=ticket_id,
                    action="set_priority",
                    reward=r,
                )

            if done:
                break
    except Exception as exc:
        print(f"Task {task} failed: {exc}", file=sys.stderr, flush=True)
        if not started:
            emit_block("START", task=task, tickets=tickets)
        score = _to_open_score(total_reward / max(1, step_count))
        emit_block(
            "END",
            task=task,
            score=score,
            steps=step_count,
            failed=True,
            error=_compact_error(exc),
        )
        return score

    score = _to_open_score(total_reward / max(1, step_count))
    emit_block("END", task=task, score=score, steps=step_count, failed=False)
    return score


def main():
    tasks = ("easy", "medium", "hard")

    try:
        client = maybe_client()
    except Exception as exc:
        print(f"Client initialization failed: {exc}", file=sys.stderr, flush=True)
        for task in tasks:
            emit_block("START", task=task, tickets=0)
            emit_block(
                "END",
                task=task,
                score=SCORE_EPSILON,
                steps=0,
                failed=True,
                error=_compact_error(exc),
            )
        raise

    env = SupportOpsEnv(seed=42)
    for task in tasks:
        run_task(env, task, client)


if __name__ == "__main__":
    main()
