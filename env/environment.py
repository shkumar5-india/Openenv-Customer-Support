import time
import uuid
from typing import Any, Dict, List, Optional
from env.graders import grade
from env.models import Action, Observation, StepResult
from env.tasks import TASKS, get_task_instructions, sample_ticket
class CustomerSupportEnv:
    def __init__(self) -> None:
        self._task_name: Optional[str] = None
        self._ticket: Optional[Dict[str, Any]] = None
        self._step_count: int = 0
        self._max_steps: int = 1
        self._done: bool = True
        self._history: List[Dict[str, Any]] = []
        self._cumulative_reward: float = 0.0
        self._episode_id: Optional[str] = None
        self._started_at: Optional[float] = None
    async def reset(self, task_name: str = "easy_classification") -> Observation:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}")
        task_def = TASKS[task_name]
        seed = int(time.time()) % 10_000
        ticket = sample_ticket(task_name, seed=seed)
        self._task_name = task_name
        self._ticket = ticket
        self._step_count = 0
        self._max_steps = task_def.max_steps
        self._done = False
        self._history = []
        self._cumulative_reward = 0.0
        self._episode_id = str(uuid.uuid4())
        self._started_at = time.time()
        return self._build_observation()
    async def step(self, action: Action) -> StepResult:
        if self._done or self._ticket is None:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"error": "Episode is done. Call reset() first."},
                error="Episode is done. Call reset() first.",
            )
        self._step_count += 1
        try:
            reward, details = grade(action, self._task_name, self._ticket)
        except Exception as exc:
            reward = 0.0
            details = {"grader_error": str(exc)}
        self._cumulative_reward += reward
        self._history.append({
            "step": self._step_count,
            "action": action.model_dump(),
            "reward": reward,
            "details": details,
        })
        done = self._step_count >= self._max_steps
        self._done = done
        next_obs = self._build_observation()
        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info={
                "step": self._step_count,
                "max_steps": self._max_steps,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "grader_details": details,
                "episode_id": self._episode_id,
                "task": self._task_name,
                "ticket_id": self._ticket.get("ticket_id"),
                "true_category": self._ticket.get("category"),
            },
            error=None,
        )
    async def close(self) -> None:
        self._done = True
        self._ticket = None
        self._task_name = None
    def state(self) -> Dict[str, Any]:
        return {
            "episode_id": self._episode_id,
            "task_name": self._task_name,
            "step_count": self._step_count,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "history_length": len(self._history),
            "ticket_id": self._ticket.get("ticket_id") if self._ticket else None,
            "started_at": self._started_at,
        }
    def _build_observation(self) -> Observation:
        if self._ticket is None or self._task_name is None:
            return Observation(
                ticket_id="",
                ticket_text="",
                task_type="",
                step_number=0,
                instructions="Call reset() to start a new episode.",
            )
        task_def = TASKS[self._task_name]
        category_hint = self._ticket["category"] if task_def.provide_hint else None

        return Observation(
            ticket_id=self._ticket["ticket_id"],
            ticket_text=self._ticket["text"],
            category_hint=category_hint,
            task_type=self._task_name,
            step_number=self._step_count + 1,
            history=[{"step": h["step"], "reward": h["reward"]} for h in self._history],
            instructions=get_task_instructions(self._task_name),
            metadata={
                "priority": self._ticket.get("priority", "medium"),
                "sentiment": self._ticket.get("sentiment", "neutral"),
                "max_steps": self._max_steps,
                "episode_id": self._episode_id,
            },
        )