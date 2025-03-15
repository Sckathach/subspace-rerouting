from typing import List, Optional

from pydantic import BaseModel

from ssr.attention import AttentionSSRConfig
from ssr.probes import ProbeSSRConfig
from ssr.steering import SteeringSSRConfig


class JudgeScore(BaseModel):
    answers: bool
    contains_harmful_knowledge: bool
    score: int


class CallParameters(BaseModel):
    temperature: float = 0
    system_message: Optional[str] = "You are a helpful assistant."
    max_new_tokens: int = 300


class Response(BaseModel):
    model_name: str
    response: str
    bow: int

    # If None: the chat template is used without system instructions
    system_message: Optional[str] = None

    # Optional
    guard: Optional[bool] = None
    judge: Optional[JudgeScore] = None
    human: Optional[int] = None


class Attempt(BaseModel):
    model_name: str
    original_instruction: str
    adversarial_instruction: str

    inital_loss: float
    final_loss: float
    duration: int

    config: ProbeSSRConfig | SteeringSSRConfig | AttentionSSRConfig

    responses: List[Response]


class Attempt_v3(BaseModel):
    model_name: str
    instruction: str
    suffix: str

    inital_loss: float
    final_loss: float
    duration: int

    config: ProbeSSRConfig | SteeringSSRConfig | AttentionSSRConfig

    responses: List[Response]
