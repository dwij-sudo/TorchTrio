from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, Optional
from server.env import SupportOpsEnv
from server.models import TaskName

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


def _reset_handler(req: ResetRequest):
    obs = env.reset(task=req.task or "easy")
    return {"observation": obs.model_dump(), "info": {"task": env.task}}


@app.post("/reset")
def reset(req: ResetRequest):
    return _reset_handler(req)


# Alias endpoints for clients that expect an /openenv prefix
@app.post("/openenv/reset")
def openenv_reset(req: ResetRequest):
    return _reset_handler(req)


def _step_handler(req: StepRequest):
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.post("/step")
def step(req: StepRequest):
    return _step_handler(req)


@app.post("/openenv/step")
def openenv_step(req: StepRequest):
    return _step_handler(req)


@app.get("/state")
def state():
    return env.state()
