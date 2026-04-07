import os
import json
from typing import Dict, Any
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback for environments without openai installed
    OpenAI = None
from server.env import SupportOpsEnv

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")  # reserved for gated deployments


def maybe_client():
    if not API_BASE_URL:
        return None
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", ""))
    if not api_key:
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=api_key)
    except Exception as e:
        print(f"Failed to create OpenAI client: {e}")
        return None


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
        print(f"LLM request failed: {e}")
        return fallback


def run_task(env: SupportOpsEnv, task: str, client) -> float:
    obs = env.reset(task=task)
    total_reward = 0.0
    tickets = len(env.tickets)

    for _ in range(tickets):
        ticket_id = env.tickets[env.current_index]["id"]

        # classification
        category = heuristic_classify(obs.ticket_text)
        obs, r, done, _ = env.step({"action_type": "classify_ticket", "category": category})
        total_reward += r

        # response
        if task in ("medium", "hard"):
            reply = llm_response(client, obs.ticket_text, obs.customer_sentiment, task)
            obs, r, done, _ = env.step({"action_type": "respond_ticket", "text": reply})
            total_reward += r

        # escalation and priority for hard
        if task == "hard":
            text_lower = obs.ticket_text.lower()
            if "outage" in text_lower or "blank screen" in text_lower:
                esc = "tier3"
                pri = "urgent"
            elif "security" in text_lower:
                esc = "tier3"
                pri = "urgent"
            elif "error" in text_lower or "crash" in text_lower or "504" in text_lower:
                esc = "tier2"
                pri = "high"
            else:
                esc = "none"
                pri = "medium"
            obs, r, done, _ = env.step({"action_type": "escalate_ticket", "level": esc})
            total_reward += r
            obs, r, done, _ = env.step({"action_type": "set_priority", "level": pri})
            total_reward += r

        if done:
            break

    return total_reward / max(1, tickets)


def main():
    client = maybe_client()
    env = SupportOpsEnv(seed=42)
    scores: Dict[str, Any] = {}
    for task in ("easy", "medium", "hard"):
        scores[task] = run_task(env, task, client)
    print(json.dumps({"scores": scores}, indent=2))


if __name__ == "__main__":
    main()
