from typing import List, Literal, Optional

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
    vanilla_input: str
    adversarial_input: str
    system_message: Optional[str] = None

    # ProbeSSR and SteeringSSR have the same config parameters, it is still possible to distinguish
    # SteeringSSR from ProbeSSR with the initial loss and final loss, but it's kinda black magic
    ssr_implementation: Literal[
        "probe",
        "steering",
        "attention_dazzle",
        "attention_dazzle_score",
        "attention_ablation",
        "attention_full_ablation",
    ]

    inital_loss: float
    final_loss: float
    duration: int

    config: ProbeSSRConfig | SteeringSSRConfig | AttentionSSRConfig

    responses: List[Response]
