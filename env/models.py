from typing import Any,Dict,List,Optional
from pydantic import BaseModel,Field
class Observation(BaseModel):
    ticket_id: str = Field(..., description="Unique ID for the support request")
    ticket_text: str = Field(..., description="The content of the customer message")
    category_hint: Optional[str] = Field(
        None, 
        description="Optional guidance for categorization"
    )
    task_type: str = Field(..., description="The specific objective type")
    step_number: int = Field(default=1, description="Sequential step index")
    history: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Log of prior interactions and outcomes"
    )
    instructions: str = Field(
        default="", 
        description="Directives for the agent"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Secondary data attributes"
    )
class Action(BaseModel):
    response_text: str = Field(
        ..., 
        description="The drafted reply for the user"
    )
    predicted_category: Optional[str] = Field(
        None, 
        description="The assigned label for the request"
    )
    should_escalate: Optional[bool] = Field(
        None, 
        description="Flag for internal transfer"
    )
    confidence: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Probability of accuracy"
    )
class StepResult(BaseModel):
    observation: Observation = Field(..., description="Resulting state post-action")
    reward: float = Field(..., ge=0.0, le=1.0, description="Evaluative metric")
    done: bool = Field(..., description="Completion status of the session")
    info: Dict[str, Any] = Field(
        default_factory=list, 
        description="Extended diagnostic data"
    )
    error: Optional[str] = Field(
        None, 
        description="Exception details if applicable"
    )
class TaskInfo(BaseModel):
    name: str
    description: str
    difficulty: str
    max_steps: int
    scoring_criteria: List[str]