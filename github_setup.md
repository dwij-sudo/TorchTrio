# GitHub Setup Guide

## Initialize
```bash
git init
git add .
git commit -m "Add SupportOpsEnv"
```

## Remote
```bash
git remote add origin <your-repo-url>
git push -u origin main
```

## Branching
- Default branch: `main`
- Features: `feat/<short-desc>`
- Fixes: `fix/<short-desc>`

## Folder structure
- `app.py`: FastAPI app for OpenEnv endpoints
- `server/`: environment logic, models, graders, rewards
- `data/`: labeled tickets dataset
- `openenv.yaml`: OpenEnv metadata
- `inference.py`: baseline runner
- `Dockerfile`: container entry

## Tips
- Keep README updated with baseline scores and deployment notes.
- Protect main with PR reviews if collaborating.
