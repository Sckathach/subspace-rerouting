import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.theme import Theme
from rich.traceback import install

install(show_locals=False)
load_dotenv()

DEVICE = "cuda:0"
DEBUG_SSR_DEFAULT = True

PROJECT_PATH = Path(__file__, "..").resolve()
PROJECT_NAME = "ssr"
DATASETS_PATH = os.getenv(
    "DATASETS_PATH", Path(PROJECT_PATH / ".." / "datasets").resolve()
)
MODELS_PATH = os.getenv(
    "MODELS_PATH", Path(PROJECT_PATH / ".." / "models.toml").resolve()
)
PROBES_WEIGTHS_PATH = os.getenv(
    "PROBES_WEIGHTS", Path(PROJECT_PATH / "probes" / "weights").resolve()
)
PROBES_CONFIG_PATH = os.getenv(
    "PROBES_CONFIG_PATH", Path(PROJECT_PATH / "probes" / "probes_config.json").resolve()
)
REFUSAL_DIRECTIONS_PATH = os.getenv(
    "REFUSAL_DIRECTIONS_PATH", Path(PROJECT_PATH / "steering" / "weights").resolve()
)
TEMPLATES_PATH = os.getenv("TEMPLATES_PATH", Path(PROJECT_PATH / "templates").resolve())


ORANGE = "#FFB84C"
PINK = "#F266AB"
VIOLET = "#A459D1"
TURQUOISE = "#2CD3E1"


custom_theme = Theme(
    {
        "orange": ORANGE,
        "violet": VIOLET,
        "turquoise": TURQUOISE,
        "pink": PINK,
    }
)
console = Console(theme=custom_theme)

pprint = console.print
