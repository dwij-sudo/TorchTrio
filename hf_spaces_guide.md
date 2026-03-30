# Hugging Face Spaces Deployment (Docker SDK)

1) Create a Space
- Go to https://huggingface.co/new-space
- Choose **Docker** SDK
- Name it (e.g., `supportopsenv`), select visibility (public/private).

2) Upload project files
- Push this repository contents (including `Dockerfile`, `requirements.txt`, `app.py`, `openenv.yaml`, `data/`, `server/`).

3) Set environment variables (Settings → Variables) — copy/paste
```
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=sk-REPLACE_ME
API_KEY=
HF_TOKEN=
PORT=7860
```
Notes:
- Use either OPENAI_API_KEY or API_KEY depending on your provider.
- HF_TOKEN only if your endpoint/model is gated via HF; otherwise leave blank.

4) Build & run
- The Space auto-builds using the Dockerfile and starts uvicorn on port 7860:
```
uvicorn app:app --host 0.0.0.0 --port 7860
```

5) Test endpoints
- Reset: `curl -X POST <space-url>/reset -H "Content-Type: application/json" -d '{"task":"medium"}'`
- Step: `curl -X POST <space-url>/step -H "Content-Type: application/json" -d '{"action":{"action_type":"classify_ticket","category":"billing"}}'`
- State: `curl <space-url>/state`

6) Troubleshooting
- Verify env vars are set.
- Check Space logs for uvicorn output.
- Ensure OpenAI-compatible endpoint is reachable from the Space.
