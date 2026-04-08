---
title: SupportOpsEnv
emoji: "📮"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SupportOpsEnv — AI Customer Support Inbox Simulation

Production-grade OpenEnv environment that mirrors a SaaS customer support inbox. Agents must classify tickets, draft empathetic responses, escalate critical issues, and set priority correctly. Three graded tasks (easy/medium/hard) with deterministic scoring and shaped rewards.

## Real-world motivation
- Reflects workflows used by support teams handling billing, technical incidents, and general inquiries.
- Evaluates LLM agents on structured decision-making, not just text generation.
- Deterministic graders ensure reproducible leaderboard-style evaluation.

## Observation space
- `ticket_text` (str): customer message.
- `customer_sentiment` (str): `positive | neutral | negative | angry`.
- `previous_actions` (List[str]): textual history for the current ticket.
- `current_task` (str): `easy | medium | hard`.

## Action space
- `classify_ticket`: `category` in {`billing`, `technical`, `general`}.
- `respond_ticket`: `text` (freeform response).
- `escalate_ticket`: `level` in {`none`, `tier1`, `tier2`, `tier3`}.
- `set_priority`: `level` in {`low`, `medium`, `high`, `urgent`}.

## Tasks
- **Easy**: classify each ticket.
- **Medium**: classify and respond with keyword/tone coverage.
- **Hard**: full workflow—classify, respond, escalate when required, set priority.

## Reward logic
- Partial rewards per action; aggregated and clamped to [-1, 1].
- Penalties: wrong classification, invalid actions, unnecessary escalation, missing required escalation/priority at ticket end.
- Response grading: keyword coverage + tone (`sorry|apologize|please|thanks`).

## Deterministic graders
- Classification: exact match.
- Response: keyword coverage + tone check.
- Escalation: exact level match; unnecessary escalation penalized.
- Priority: exact level match.

## Dataset
Realistic tickets stored in `data/tickets.json` with sentiment, expected category, keywords, escalation, and priority labels.

## Setup
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Example (Python)
```python
from server.env import SupportOpsEnv
env = SupportOpsEnv(seed=7)
obs = env.reset(task="medium")
obs, reward, done, info = env.step({"action_type": "classify_ticket", "category": "billing"})
```

## Baseline runner
```bash
python inference.py
```
Outputs JSON scores for easy/medium/hard using heuristic classification and optional OpenAI responses (`API_BASE_URL`, `API_KEY`, `MODEL_NAME`, `HF_TOKEN`).
Defaults: `API_BASE_URL=https://api.openai.com/v1`, `MODEL_NAME=gpt-4.1-mini`. Set `HF_TOKEN` manually as a secret/environment variable.

## Docker
```bash
docker build -t supportopsenv .
docker run -p 7860:7860 supportopsenv
```

## Hugging Face Spaces
- Use Docker SDK, set env vars `API_BASE_URL`, `API_KEY`, `MODEL_NAME`, `HF_TOKEN`.
- See `hf_spaces_guide.md` for deployment steps and endpoint testing.

## OpenEnv metadata
See `openenv.yaml` (entrypoint `server.env:SupportOpsEnv`) for task list and spec fields.
