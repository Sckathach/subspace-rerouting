import json

from ssr.lens import Lens
from ssr.probes import PROBES_CONFIG_PATH, PROBES_PATH
from ssr.probes.classifiers import init_probes

MODEL_NAME = "qwen2.5_1.5b"
lens = Lens.from_config(MODEL_NAME)


with open(PROBES_CONFIG_PATH, "r") as file:
    probes_config = json.load(file)

probes_config[MODEL_NAME]

probes_config_values = probes_config[MODEL_NAME].values()

init_probes(
    lens=lens,
    model_name=MODEL_NAME,
    loss_names=[x["loss_name"] for x in probes_config_values],
    optimizer_names=[x["optimizer"] for x in probes_config_values],
    lrs=[x["lr"] for x in probes_config_values],
    epochs=[x["epochs"] for x in probes_config_values],
    save_directory=f"{PROBES_PATH}/probes/{MODEL_NAME}/",
    verbose=True,
)
