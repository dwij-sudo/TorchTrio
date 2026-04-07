from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, Optional
from .env import SupportOpsEnv
from .models import TaskName

app = FastAPI(title="SupportOpsEnv", version="1.0.0")
env = SupportOpsEnv()


class ResetRequest(BaseModel):
    # Accept both "task" and "task_id" payload keys to match OpenEnv clients
    model_config = ConfigDict(populate_by_name=True)

    task: Optional[TaskName] = Field(default="easy", alias="task_id")


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/")
def root():
    return {
        "name": "SupportOpsEnv",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "docs": "/docs",
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    task = req.task if req and req.task else "easy"
    obs = env.reset(task=task)
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(req: StepRequest):
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()