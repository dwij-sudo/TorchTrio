---
title: SupportOpsEnv
emoji: "📮"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SupportOpsEnv — TorchTrio

SupportOpsEnv — TorchTrio is a realistic SaaS customer support inbox Mini-RL environment built on OpenEnv.
It evaluates whether an agent can make reliable support decisions end-to-end: classify tickets, craft empathetic responses, escalate when needed, and assign the right priority.

The environment is intentionally deterministic and reproducible, making it suitable for leaderboard-style evaluation and hackathon judging.

## Why this matters
- Mirrors production support operations across billing, technical incidents, and general customer communication.
- Tests structured decision quality, not only text fluency.
- Encourages policy consistency under realistic constraints (sentiment, escalation risk, and action history).

## Real-world impact and business value
- Faster resolution loops: better first-pass classification and priority assignment can reduce mean resolution time by routing tickets correctly earlier.
- Better escalation accuracy: penalizing unnecessary escalation while requiring escalation for critical tickets improves tier utilization and incident response quality.
- Higher customer trust: response grading rewards empathetic tone and concrete next-step communication.
- Reproducible evaluation: deterministic graders remove ambiguity and make progress measurable across model versions.

## Environment design
### Observation space
- `ticket_text` (str): raw customer message.
- `customer_sentiment` (str): `positive | neutral | negative | angry`.
- `previous_actions` (List[str]): action trace for the current ticket.
- `current_task` (str): `easy | medium | hard`.

### Action space
- `classify_ticket`: `category` in {`billing`, `technical`, `general`}.
- `respond_ticket`: `text` (freeform response).
- `escalate_ticket`: `level` in {`none`, `tier1`, `tier2`, `tier3`}.
- `set_priority`: `level` in {`low`, `medium`, `high`, `urgent`}.

## Tasks
- **Easy**: ticket classification.
- **Medium**: classification + response quality.
- **Hard**: full workflow with classification, response, escalation, and priority.

## Reward shaping and deterministic grading
- Partial rewards are weighted by action type and clamped to [-1, 1].
- Terminal penalties handle missing required escalation/priority and poor workflow completion.
- Deterministic grader for classification: exact label match.
- Deterministic grader for response: keyword coverage + empathetic tone check (`sorry|apologize|please|thanks`).
- Deterministic grader for escalation: exact level match with unnecessary escalation penalties.
- Deterministic grader for priority: exact level match.

## Dataset
Realistic labeled tickets live in `data/tickets.json` with sentiment, expected category, response keywords, escalation level, and priority.

## Model compatibility
- Works with OpenAI-compatible APIs through `API_BASE_URL` + `MODEL_NAME` + `HF_TOKEN`.
- Compatible with Llama-family models served from Hugging Face Inference Endpoints / TGI / vLLM, as long as they expose an OpenAI-compatible interface.
- Default `API_BASE_URL=https://api.openai.com/v1`.
- Default `MODEL_NAME=gpt-4.1-mini`.

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
The baseline emits per-task `[START]`, `[STEP]`, and `[END]` blocks for `easy`, `medium`, and `hard`.
`HF_TOKEN` is required and is used as the OpenAI client `api_key`.

## Docker
```bash
docker build -t supportopsenv .
docker run -p 7860:7860 supportopsenv
```

## Hugging Face Spaces
- Use Docker SDK.
- Set environment variables: `API_BASE_URL`, `MODEL_NAME`, `PORT=7860`.
- Set secret: `HF_TOKEN`.
- Full deployment guide: `hf_spaces_guide.md`.

## OpenEnv metadata
See `openenv.yaml` (entrypoint `server.env:SupportOpsEnv`) for task metadata and environment schema.

## Grand Finale Extensions
Potential extensions for the Bangalore finale without changing the current core environment contract:
- Procedural ticket generation with controlled domain shifts (product updates, outage bursts, seasonal billing spikes).
- TRL/PPO training example using this environment for policy optimization beyond prompt-only baselines.
- Multi-ticket batching to benchmark throughput and queue-level decision quality under realistic support load.
