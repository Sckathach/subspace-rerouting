import datetime
import json
import os
from pathlib import Path
from typing import Any, List, Literal, Tuple, overload

import torch as t
from jinja2 import Environment, PackageLoader, Template

from ssr import PROJECT_NAME, TEMPLATES_PATH


@overload
def get_template(path: str) -> Template: ...


@overload
def get_template(path: str, str_output: bool) -> str: ...


def get_template(path: str, str_output: bool = False) -> Template | str:
    if ".jinja2" not in path:
        path += ".jinja2"

    if str_output:
        with open(TEMPLATES_PATH / Path(path), "r") as f:
            return f.read()

    env = Environment(loader=PackageLoader(PROJECT_NAME))
    return env.get_template(path)


def load_jsonl(filepath: str) -> List[dict]:
    data = []
    with open(filepath, "r") as f:
        for k, line in enumerate(f.readlines()):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON could not be decoded from file: {filepath} at line {k}: {line}"
                ) from e
    return data


def load_dataset(
    dataset_name: Literal["mod", "adv", "mini", "bomb"] = "mod",
    max_samples: int = 120,
) -> Tuple[List[str], List[str]]:
    match dataset_name:
        case "mod":
            with open("datasets/advbench_modified.json", "r") as f:
                dataset = json.load(f)
                hf_raw, hl_raw = (
                    [x["harmful"] for x in dataset],
                    [x["harmless"] for x in dataset],
                )

                return hf_raw[:max_samples], hl_raw[:max_samples]

        case "adv":
            with open("datasets/advbench_alpaca.json", "r") as f:
                dataset = json.load(f)
                hf_raw, hl_raw = dataset["harmful"], dataset["harmless"]

                return hf_raw[:max_samples], hl_raw[:max_samples]

        case "mini":
            with open("datasets/minibench.json", "r") as f:
                minibench = json.load(f)

                return minibench["harmful"][:max_samples], minibench["harmless"][
                    :max_samples
                ]

        case "bomb":
            with open("datasets/bomb_extended.json", "r") as f:
                bomb_extended = json.load(f)

                return bomb_extended["harmful"][:max_samples], bomb_extended[
                    "harmless"
                ][:max_samples]


def log_jsonl(filepath: str, data: dict) -> None:
    with open(filepath, "a") as f:
        f.write(json.dumps(data) + "\n")
        f.close()


def save_weights(
    directory: str,
    model_name: str,
    weights_filename: str,
    weights: Any,
    override: bool = False,
):
    save_directory = (Path(directory) / model_name).resolve()
    os.makedirs(save_directory, exist_ok=True)

    weights_path = Path(save_directory / weights_filename).resolve()

    if not override and os.path.exists(weights_path):
        raise ValueError("Weights already exist! Use override = True to write anyway.")

    t.save(weights, weights_path)


def save_metadata(
    directory: str,
    model_name: str,
    metadata: dict,
    override: bool = False,
    add_timestamp: bool = True,
):
    save_directory = (Path(directory) / model_name).resolve()
    os.makedirs(save_directory, exist_ok=True)

    metadata_path = Path(save_directory / "metadata.json").resolve()

    if not override and os.path.exists(metadata_path):
        raise ValueError("Metadata already exist! Use override = True to write anyway.")

    if add_timestamp:
        metadata = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        } | metadata

    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)


def load_weights(directory: str, model_name: str, weights_filename: str):
    weights_path = (Path(directory) / model_name / weights_filename).resolve()
    if not os.path.exists(weights_path):
        raise ValueError(f"File {weights_path} does not exist!")

    return t.load(weights_path)


def load_metadata(directory: str, model_name: str):
    metadata_path = (Path(directory) / model_name / "metadata.json").resolve()
    if not os.path.exists(metadata_path):
        raise ValueError(f"File {metadata_path} does not exist!")

    with open(metadata_path, "r") as metadata_file:
        return json.load(metadata_file)
