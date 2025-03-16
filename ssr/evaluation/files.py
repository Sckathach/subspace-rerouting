import json
from typing import List

from rich import print

from ssr.evaluation.types import Attempt


def load_attempts_jsonl(file_path: str) -> List[Attempt]:
    with open(file_path, "r") as f:
        lines = f.readlines()

    d_lines: List[Attempt] = []
    for line in lines:
        try:
            d_lines.append(Attempt(**json.loads(line)))
        except Exception as e:
            print(line, e)

    return d_lines
