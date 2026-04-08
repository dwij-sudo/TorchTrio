# Hugging Face Spaces Deployment (Docker SDK)

1) Create a Space
- Go to https://huggingface.co/new-space
- Choose **Docker** SDK
- Name it (e.g., `supportopsenv`), select visibility (public/private).

2) Upload project files
- Push this repository contents (including `Dockerfile`, `requirements.txt`, `app.py`, `openenv.yaml`, `data/`, `server/`).

3) Set environment variables (Settings → Variables)
```
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4.1-mini
PORT=7860
```
Set this in Settings → Secrets:
```
HF_TOKEN=hf_xxx
```
Notes:
- `API_BASE_URL` defaults to `https://api.openai.com/v1` if unset.
- `MODEL_NAME` defaults to `gpt-4.1-mini` if unset.
- `HF_TOKEN` is required and is used as `api_key` for the OpenAI client.
- Do not hardcode credentials in code.

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
