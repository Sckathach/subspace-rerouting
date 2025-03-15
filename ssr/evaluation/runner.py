import time
from functools import partial
from typing import List, Literal, Optional, cast

import torch as t
from pydantic import BaseModel
from rich import print

from ssr.evaluation.api import call_guard, call_lmstudio, call_ollama
from ssr.evaluation.quick_eval import rcall_judge
from ssr.evaluation.types import Attempt, JudgeScore, Response
from ssr.evaluation.utils import harmful_bow
from ssr.files import load_dataset, log_jsonl
from ssr.lens import Lens
from ssr.probes import ProbeSSR
from ssr.steering import SteeringSSR


class RunnerConfig(BaseModel):
    log_filename: str
    instruction_dataset: Literal["mini", "adv", "mod", "bomb"] = "mini"
    mask: str = "{instruction} [MASK][MASK][MASK][MASK]"
    target_api: Literal["ollama", "lmstudio", "none"] = "lmstudio"
    verbose: bool = False


class Runner:
    def __init__(
        self, lens: Lens, ssr: ProbeSSR | SteeringSSR, runner_config: RunnerConfig
    ):
        self.lens = lens
        self.ssr = ssr
        self.run_config = runner_config

        self.instructions, _ = load_dataset(runner_config.instruction_dataset)

        self.current_duration: int
        self.current_candidates: List[str]
        self.current_losses: List[float]
        self.current_instruction: Optional[str]
        self.current_candidate: Optional[str]
        self.current_loss: Optional[float]
        self.current_response: Optional[str]
        self.current_judge: Optional[JudgeScore]
        self.current_guard: Optional[bool]
        self.current_bow: Optional[int]
        self.chat_template_before: str
        self.chat_template_after: str

        self.reset()

        if self.run_config.target_api == "ollama":
            self.call_api = partial(
                call_ollama,
                model_name=self.ssr.config.model_name.replace("_", ":"),
                system_message=self.lens.default_values.system_message,
                verbose=self.run_config.verbose,
            )
        elif self.run_config.target_api == "lmstudio":
            self.call_api = partial(
                call_lmstudio,
                model_name=self.ssr.config.model_name,
                system_message=self.lens.default_values.system_message,
                verbose=self.run_config.verbose,
            )

    def reset_chat_template(self) -> None:
        self.chat_template_before, self.chat_template_after = (
            self.lens.apply_chat_template(
                "[CROISSANT]", system_message=self.lens.default_values.system_message
            ).split("[CROISSANT]")
        )

    def reset_current_generation(self) -> None:
        self.current_duration = 666
        self.current_candidates = []
        self.current_losses = []
        self.current_instruction = None

    def reset_current_attempt(self) -> None:
        self.current_candidate = None
        self.current_loss = None
        self.current_response = None
        self.current_judge = None
        self.current_guard = None
        self.current_bow = None

    def reset(self) -> None:
        self.reset_current_attempt()
        self.reset_current_generation()
        self.reset_chat_template()

    def extract_instruction(self, instruction_with_chat_template: str) -> str:
        try:
            return instruction_with_chat_template.split(self.chat_template_before)[
                1
            ].split(self.chat_template_after)[0]
        except:  # noqa: E722
            print(f"""ERROR extracting the instruction. 
                Before chat template: {self.chat_template_before}
                After chat template: {self.chat_template_after}
                Instruction: {instruction_with_chat_template}
                """)
            exit(0)

    def generate(self) -> None:
        if self.current_instruction is None:
            raise ValueError("current_instruction is None")

        start = time.time()
        self.ssr.init_prompt(
            self.lens.apply_chat_template(
                self.run_config.mask.format(instruction=self.current_instruction),
                system_message=self.lens.default_values.system_message,
            )
        )
        self.ssr.buffer_init_random()
        self.ssr.generate()
        self.current_duration = int(time.time() - start)

    def extract_buffers(self, reset_run_buffers: bool = True) -> None:
        if reset_run_buffers:
            self.candidates = []
            self.losses = []

        for ids, loss in zip(
            t.cat([self.ssr.candidate_ids.cpu(), self.ssr.archive_ids], dim=0),
            self.ssr.candidate_losses.tolist() + self.ssr.archive_losses.tolist(),
        ):
            self.candidates.append(
                self.extract_instruction(
                    self.lens.model.to_string(self.ssr.get_tokens(ids.unsqueeze(0)))[0]
                )
            )
            self.losses.append(loss)

    def call_judge(self) -> None:
        if self.current_instruction is None:
            raise ValueError("current_instruction is None")
        elif self.current_response is None:
            raise ValueError("current_response is None")
        self.current_judge = rcall_judge(
            self.current_instruction, self.current_response
        )

    def call_guard(self) -> None:
        if self.current_response is None:
            raise ValueError("current_response is None")
        self.current_guard = call_guard(self.current_response)

    def call_harmful_bow(self) -> None:
        if self.current_response is None:
            raise ValueError("current_response is None")
        self.current_bow = harmful_bow(self.current_response)

    def log_attempt(self) -> None:
        stack_value_errors = ""
        if self.current_instruction is None:
            stack_value_errors += "\n - current_instruction"
        if self.current_candidate is None:
            stack_value_errors += "\n - current_candidate"
        if self.current_loss is None:
            stack_value_errors += "\n - current_loss"
        if self.current_duration is None:
            stack_value_errors += "\n - current_duration"
        if len(stack_value_errors) > 1:
            raise ValueError(f"Following values are None: {stack_value_errors}")

        responses = []
        if self.current_response is not None:
            responses.append(
                Response(
                    model_name=self.ssr.config.model_name,
                    response=self.current_response,
                    system_message=self.lens.default_values.system_message,
                    bow=harmful_bow(self.current_response),
                    guard=self.current_guard,
                    judge=self.current_judge,
                )
            )
        log_jsonl(
            self.run_config.log_filename,
            Attempt(
                model_name=self.ssr.config.model_name,
                original_instruction=cast(str, self.current_instruction),
                adversarial_instruction=cast(str, self.current_candidate),
                inital_loss=self.ssr.initial_loss,
                final_loss=cast(float, self.current_loss),
                duration=cast(int, self.current_duration),
                config=self.ssr.config,
                responses=responses,
            ).model_dump(),
        )
