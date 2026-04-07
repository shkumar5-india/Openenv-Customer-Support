import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_SOURCE = os.path.join(_SCRIPT_DIR, "..", "data", "support_tickets.csv")
@dataclass
class TaskDefinition:
    name: str
    description: str
    difficulty: str
    max_steps: int
    scoring_criteria: List[str]
    category_filter: Optional[str] = None
    provide_hint: bool = False
TASKS: Dict[str, TaskDefinition] = {
    "easy_classification": TaskDefinition(
        name="easy_classification",
        description=(
            "Identify the ticket category and provide a polite confirmation. "
            "A hint is included to assist with classification."
        ),
        difficulty="easy",
        max_steps=1,
        scoring_criteria=["category_match", "politeness", "response_quality"],
        provide_hint=True,
    ),
    "refund_handling": TaskDefinition(
        name="refund_handling",
        description=(
            "Manage a refund request by validating eligibility and "
            "explaining the processing timeline to the user."
        ),
        difficulty="medium",
        max_steps=2,
        scoring_criteria=["category_match", "politeness", "response_quality", "refund_clarity"],
        category_filter="refund",
    ),
    "escalation_decision": TaskDefinition(
        name="escalation_decision",
        description=(
            "Determine if a sensitive or high-priority issue requires "
            "human intervention based on specific risk triggers."
        ),
        difficulty="hard",
        max_steps=2,
        scoring_criteria=["category_match", "politeness", "response_quality", "escalation_accuracy"],
    ),
}
_CACHE: Optional[pd.DataFrame] = None
def _fetch_records(limit: int = 500, entropy: int = 42) -> pd.DataFrame:
    df = pd.read_csv(_DATA_SOURCE)
    df = df.dropna(subset=["text", "category"])
    df["category"] = df["category"].str.lower().str.strip()  
    if len(df) > limit:
        df = df.sample(n=limit, random_state=entropy).reset_index(drop=True)
    return df
def get_dataset() -> pd.DataFrame:
    global _CACHE
    if _CACHE is None:
        _CACHE = _fetch_records()
    return _CACHE
def sample_ticket(task_name: str, seed: Optional[int] = None) -> Dict[str, Any]:
    config = TASKS[task_name]
    data = get_dataset()
    if config.category_filter:
        pool = data[data["category"] == config.category_filter]
        if pool.empty:
            pool = data
    elif task_name == "escalation_decision":
        pool = data[(data["priority"] == "high") | (data["category"] == "complaint")]
        if pool.empty:
            pool = data
    else:
        pool = data
    generator = random.Random(seed)
    row_idx = generator.randint(0, len(pool) - 1)
    selection = pool.iloc[row_idx]
    return {
        "ticket_id": str(selection.get("ticket_id", f"T{row_idx:04d}")),
        "text": str(selection["text"]),
        "category": str(selection["category"]),
        "priority": str(selection.get("priority", "medium")),
        "sentiment": str(selection.get("sentiment", "neutral")),
    }
def get_task_instructions(task_name: str) -> str:
    manifest = {
        "easy_classification": (
            "Analyze the support request. 1) Identify the category (delivery, refund, complaint). "
            "2) Draft a courteous 2-4 sentence acknowledgment. A hint is available."
        ),
        "refund_handling": (
            "Process a refund inquiry. 1) Label as 'refund'. 2) Draft a 3-5 sentence reply "
            "addressing eligibility and the expected resolution window. 3) Escalate only if critical."
        ),
        "escalation_decision": (
            "Evaluate for human oversight. 1) Categorize the ticket. 2) Provide an empathetic reply. "
            "3) Enable escalation if the item is damaged, the user reports staff misconduct, "
            "the issue persists across multiple attempts, or high-priority negative sentiment exists."
        ),
    }
    return manifest.get(task_name, "Provide a helpful and professional response.")
def list_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "name": obj.name,
            "description": obj.description,
            "difficulty": obj.difficulty,
            "max_steps": obj.max_steps,
            "scoring_criteria": obj.scoring_criteria,
        }
        for obj in TASKS.values()
    ]