import os
from pathlib import Path

from ssr.probes.main import ProbeSSR, ProbeSSRConfig

PROBES_PATH = Path(__file__, "..").resolve()
PROBES_CONFIG_PATH = os.getenv(
    "PROBES_CONFIG_PATH", Path(PROBES_PATH / "probes_config.json").resolve()
)
