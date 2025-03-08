import argparse
import contextlib
import json
import os

import nanogcg
import toml
import torch as t
from nanogcg import GCGConfig
from pydantic import BaseModel
from rich import print
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

parser = argparse.ArgumentParser(description="Run GCG attack on language models")
parser.add_argument("-c", "--config", help="Path to the config file")
parser.add_argument("--model_name", type=str, help="Model identifier to use")
parser.add_argument("--display_name", type=str, help="Display name for output files")
parser.add_argument("--num_steps", type=int, help="Number of steps")
parser.add_argument("--search_width", type=int, help="Search width")
parser.add_argument("--topk", type=int, help="Top k")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--temperature", type=float, help="Temperature for generation")
parser.add_argument("--separator", type=str, help="Separator token")
parser.add_argument("--optim_str_init", type=str, help="Initial optimization string")
parser.add_argument("--dataset", type=str, help="Path to the dataset file")
parser.add_argument("--output_dir", type=str, help="Directory for output files")

args = parser.parse_args()


# Default configuration
class ExperimentConfig(BaseModel):
    model_name: str
    num_steps: int = 500
    search_width: int = 512
    topk: int = 64
    batch_size: int = 62
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    temperature: float = 0.1
    separator: str = "assistant"
    device: str = "cuda"
    n_responses: int = 3
    max_new_tokens: int = 150
    trust_remote_code: bool = False
    local_files_only: bool = True
    dataset: str = "datasets/minibench.json"
    output_dir: str = "reproduce_experiments/gcg/results"


# Load from config file if provided
if args.config:
    with open(args.config, "r") as f:
        file_config = toml.load(f)
        config = ExperimentConfig(**file_config)
else:
    base_config = ExperimentConfig(model_name=args.model_name).model_dump()

    config = ExperimentConfig(
        **(base_config | {k: v for k, v in vars(args).items() if v is not None})
    )


print(f"Config taken into account:\n{config}")

"""Set up the model and tokenizer based on configuration."""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=t.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    trust_remote_code=config.trust_remote_code,
    local_files_only=config.local_files_only,
)

model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=t.float16,
    trust_remote_code=config.trust_remote_code,
    quantization_config=bnb_config,
    local_files_only=config.local_files_only,
).to(config.device)

gcg_config = GCGConfig(
    num_steps=config.num_steps,
    search_width=config.search_width,
    topk=config.topk,
    use_prefix_cache=False,
    batch_size=config.batch_size,
    optim_str_init=config.optim_str_init,
    buffer_size=1,  # issues with buffer > 1
)

with open(config.dataset, "r") as f:
    data = json.load(f)
instructions = data["harmful"]
targets = data["gcg_targets"]

output_dir = config.output_dir
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{config.model_name.split('/')[-1]}.jsonl")


def append(data: dict) -> None:
    with open(output_path, "a") as f:
        f.write(json.dumps(data) + "\n")
        f.close()


def apply_chat_template(prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )  # type: ignore


def call_llm(input_text: str) -> str:
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **input_ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        use_cache=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


test = apply_chat_template("USER PROMPT")
print(test)


for instruction, target in zip(instructions, targets):
    print(f"Starting on instruction: {instruction}")

    result = nanogcg.run(model, tokenizer, instruction, target, gcg_config)  # type: ignore

    for _ in range(config.n_responses):
        adv_prompt = apply_chat_template(instruction + result.best_string)
        response = call_llm(adv_prompt)

        with contextlib.suppress(IndexError):
            response = response.split(config.separator)[-1].strip(config.separator)

        append(
            config.model_dump()
            | {
                "instruction": instruction,
                "target": target,
                "prompt": result.best_string,
                "loss": result.best_loss,
                "response": response,
                "chat_template_example": test,
            },
        )
