import json
from pathlib import Path
from typing import List, overload

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


def log_jsonl(filepath: str, data: dict) -> None:
    with open(filepath, "a") as f:
        f.write(json.dumps(data) + "\n")
        f.close()
