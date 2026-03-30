import random
from typing import Dict, List, Optional, Tuple
from pydantic import ValidationError, TypeAdapter
from .models import Observation, Action, TaskName
from . import tasks
from . import graders
from . import reward


class SupportOpsEnv:
    def __init__(self, seed: Optional[int] = None, max_steps: int = 6):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.task: TaskName = "easy"
        self.tickets: List[Dict] = []
        self.current_index: int = 0
        self.previous_actions: List[str] = []
        self.steps_taken: int = 0
        self.expected: Dict = {}
        self._action_adapter = TypeAdapter(Action)
        self.reset(task="easy")

    def reset(self, task: TaskName = "easy") -> Observation:
        self.task = task
        self.tickets = tasks.get_tickets(task, self.rng)
        self.current_index = 0
        self.previous_actions = []
        self.steps_taken = 0
        self._set_expected()
        return self._observation()

    def state(self) -> Dict:
        return {
            "task": self.task,
            "ticket_id": self.tickets[self.current_index]["id"],
            "index": self.current_index,
            "steps_taken": self.steps_taken,
            "previous_actions": list(self.previous_actions),
        }

    def _set_expected(self) -> None:
        ticket = self.tickets[self.current_index]
        self.expected = {
            "category": ticket["category"],
            "keywords": ticket.get("response_keywords", []),
            "escalation": ticket.get("escalation", "none"),
            "priority": ticket.get("priority", "medium"),
        }

    def _observation(self) -> Observation:
        ticket = self.tickets[self.current_index]
        return Observation(
            ticket_text=ticket["text"],
            customer_sentiment=ticket["sentiment"],
            previous_actions=list(self.previous_actions),
            current_task=self.task,
        )

    def step(self, action_dict: Dict) -> Tuple[Observation, float, bool, Dict]:
        try:
            action = self._action_adapter.validate_python(action_dict)
        except ValidationError as exc:
            return self._observation(), -0.2, False, {"error": exc.errors()}

        reward_value, ticket_done = self._apply_action(action)
        finished_all = False

        if ticket_done:
            finished_all = self._advance_ticket()

        obs = self._observation()
        done = finished_all
        return obs, reward_value, done, {}

    def _advance_ticket(self) -> bool:
        self.current_index += 1
        if self.current_index >= len(self.tickets):
            self.current_index = len(self.tickets) - 1
            return True
        self.previous_actions = []
        self.steps_taken = 0
        self._set_expected()
        return False

    def _apply_action(self, action: Action) -> Tuple[float, bool]:
        self.steps_taken += 1
        weighted: List[Tuple[float, float]] = []

        if action.action_type == "classify_ticket":
            score = graders.grade_classification(action.category, self.expected["category"])
            weighted.append((0.5, score))
            self.previous_actions.append(f"classify:{action.category}")
            if score == 0.0:
                weighted.append((0.3, -0.5))

        elif action.action_type == "respond_ticket":
            score = graders.grade_response(action.text, self.expected["keywords"])
            weighted.append((0.4, score))
            self.previous_actions.append("respond")

        elif action.action_type == "escalate_ticket":
            expected = self.expected["escalation"]
            score = graders.grade_escalation(action.level, expected)
            if expected == "none" and action.level != "none":
                weighted.append((0.3, -0.5))
            else:
                weighted.append((0.3, score))
            self.previous_actions.append(f"escalate:{action.level}")

        elif action.action_type == "set_priority":
            score = graders.grade_priority(action.level, self.expected["priority"])
            weighted.append((0.3, score))
            self.previous_actions.append(f"priority:{action.level}")

        ticket_done = self._is_ticket_done()

        if ticket_done:
            weighted.extend(self._terminal_penalties())

        return reward.aggregate(weighted), ticket_done

    def _is_ticket_done(self) -> bool:
        classify_done = any(pa.startswith("classify:") for pa in self.previous_actions)
        respond_done = any(pa == "respond" for pa in self.previous_actions)
        escalate_done = any(pa.startswith("escalate:") for pa in self.previous_actions)
        priority_done = any(pa.startswith("priority:") for pa in self.previous_actions)

        if self.task == "easy":
            return classify_done or self.steps_taken >= self.max_steps
        if self.task == "medium":
            return (classify_done and respond_done) or self.steps_taken >= self.max_steps
        return (
            classify_done
            and respond_done
            and priority_done
            and (self.expected["escalation"] == "none" or escalate_done)
        ) or self.steps_taken >= self.max_steps

    def _terminal_penalties(self) -> List[Tuple[float, float]]:
        penalties: List[Tuple[float, float]] = []
        expected_esc = self.expected["escalation"]
        expected_pri = self.expected["priority"]

        escalate_done = any(pa.startswith("escalate:") for pa in self.previous_actions)
        priority_done = any(pa.startswith("priority:") for pa in self.previous_actions)

        if self.task == "hard":
            if expected_esc != "none" and not escalate_done:
                penalties.append((0.3, -0.5))
            if not priority_done:
                penalties.append((0.3, -0.3))
        return penalties
