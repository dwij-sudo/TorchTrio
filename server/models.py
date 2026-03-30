from typing import List, Literal, Union, Annotated, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

Sentiment = Literal["positive", "neutral", "negative", "angry"]
TaskName = Literal["easy", "medium", "hard"]
Category = Literal["billing", "technical", "general"]
EscalationLevel = Literal["none", "tier1", "tier2", "tier3"]
PriorityLevel = Literal["low", "medium", "high", "urgent"]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_text: str
    customer_sentiment: Sentiment
    previous_actions: List[str]
    current_task: TaskName


class ClassifyAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["classify_ticket"] = Field(default="classify_ticket")
    category: Category


class RespondAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["respond_ticket"] = Field(default="respond_ticket")
    text: str = Field(min_length=1)


class EscalateAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["escalate_ticket"] = Field(default="escalate_ticket")
    level: EscalationLevel


class PriorityAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["set_priority"] = Field(default="set_priority")
    level: PriorityLevel


Action = Annotated[Union[ClassifyAction, RespondAction, EscalateAction, PriorityAction], Field(discriminator="action_type")]


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
