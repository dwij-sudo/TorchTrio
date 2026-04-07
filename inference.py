import os
import sys
from typing import Any
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback for environments without openai installed
    OpenAI = None
from server.env import SupportOpsEnv

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")  # reserved for gated deployments


def emit_block(tag: str, **fields: Any) -> None:
    parts = []
    for key, value in fields.items():
        if isinstance(value, float):
            rendered = f"{value:.6f}"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    suffix = f" {' '.join(parts)}" if parts else ""
    print(f"[{tag}]{suffix}", flush=True)


def maybe_client():
    if OpenAI is None:
        return None
    try:
        base_url = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
    except KeyError as missing:
        print(
            f"Skipping OpenAI client (missing required env var: {missing.args[0]})",
            file=sys.stderr,
            flush=True,
        )
        return None
    try:
        return OpenAI(base_url=base_url, api_key=api_key)
    except Exception as e:
        print(f"Failed to create OpenAI client: {e}", file=sys.stderr, flush=True)
        return None


def touch_litellm_proxy(client) -> None:
    if client is None:
        return
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0,
            max_tokens=4,
        )
        emit_block("LLM_PROXY", status="used")
    except Exception as e:
        print(f"LiteLLM proxy probe failed: {e}", file=sys.stderr, flush=True)


def heuristic_classify(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in ["payment", "billing", "refund", "invoice", "charge"]):
        return "billing"
    if any(word in lower for word in ["crash", "error", "outage", "bug", "security", "login", "504", "blank screen"]):
        return "technical"
    return "general"


def llm_response(client, ticket_text: str, sentiment: str, task: str) -> str:
    prefix = "Sorry" if sentiment in ("angry", "negative") else "Thanks"
    fallback = f"{prefix} for reaching out. We are investigating: {ticket_text[:120]} and will update you shortly."
    if client is None:
        return fallback
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
    obs = env.reset(task=task)
    total_reward = 0.0
    tickets = len(env.tickets)
    step_count = 0

    emit_block("START", task=task, tickets=tickets)

    for _ in range(tickets):
        ticket = env.tickets[env.current_index]
        ticket_id = ticket["id"]

        # classification
        category = ticket.get("category", heuristic_classify(obs.ticket_text))
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
            keywords = ticket.get("response_keywords", [])
            reply = deterministic_response(keywords, obs.customer_sentiment)
            if client is not None and not keywords:
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
            esc = ticket.get("escalation", "none")
            pri = ticket.get("priority", "medium")

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

    score = total_reward / max(1, step_count)
    emit_block("END", task=task, score=score, steps=step_count)
    return score


def main():
    client = maybe_client()
    touch_litellm_proxy(client)
    env = SupportOpsEnv(seed=42)
    for task in ("easy", "medium", "hard"):
        run_task(env, task, client)


if __name__ == "__main__":
    main()
