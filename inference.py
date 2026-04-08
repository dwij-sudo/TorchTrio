import os
import sys
from typing import Any
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback for environments without openai installed
    OpenAI = None
from server.env import SupportOpsEnv

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
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
        # Create client with only the required parameters to avoid proxy conflicts
        # Remove any environment variables that might interfere with client initialization
        client = OpenAI(base_url=base_url, api_key=api_key)
        return client
    except TypeError as e:
        # Handle cases where unexpected kwargs are being passed (e.g., proxies)
        if "proxies" in str(e) or "unexpected keyword" in str(e):
            print(f"Failed to create OpenAI client (proxies argument issue): {e}", file=sys.stderr, flush=True)
            return None
        raise
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


def llm_classify(client, ticket_text: str, sentiment: str) -> str:
    if client is None:
        return heuristic_classify(ticket_text)
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
    if client is None:
        return "none"
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
    if client is None:
        return "medium"
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
