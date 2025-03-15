import os
from typing import Dict, List, Tuple

import torch as t
import tqdm
import transformer_lens as tl
from jaxtyping import Float

from ssr.core import SSR, SSRConfig
from ssr.lens import Lens
from ssr.probes.classifiers import LinearClassifier
from ssr.types import HookList, Loss


class ProbeSSRConfig(SSRConfig):
    model_name: str
    layers: List[int]
    alphas: List[float]
    pattern: str
    load_directory: str
    system_message: str


class ProbeSSR(SSR):
    def __init__(self, lens: Lens, config: ProbeSSRConfig):
        super().__init__(lens.model, config)

        self.lens = lens
        self.config: ProbeSSRConfig = config

        self.probes: Dict[int, Tuple[t.nn.Module, float, Loss]] = {}
        self.act_dict: Dict[str, t.Tensor] = {}
        self.fwd_hooks: HookList

        self.setup_hooks()
        self.load_probes()

    def setup_hooks(self):
        # Has to be called again if config.layers have changed
        self.fwd_hooks = [
            (tl.utils.get_act_name(self.config.pattern, layer), self.hook_fn)
            for layer in self.config.layers
        ]
        self.config.max_layer = max(self.config.layers)

    def hook_fn(self, activations, hook):
        self.act_dict[hook.name] = activations.float()
        return activations

    def load_probes(self, verbose: bool = False):
        for layer, alpha in tqdm.tqdm(zip(self.config.layers, self.config.alphas)):
            probe_path = os.path.join(
                self.config.load_directory, f"{self.config.model_name}_{layer}.pt"
            )
            if not os.path.exists(probe_path):
                raise ValueError(f"ERROR: No saved probe found for layer {layer}")

            probe_info = t.load(probe_path)

            classifier = LinearClassifier(self.model.cfg.d_model).to(self.device)
            classifier.load_state_dict(probe_info["state_dict"])
            classifier.eval()

            for param in classifier.parameters():
                param.requires_grad = False

            loss_fn = (
                t.nn.MSELoss(reduction="none").to(self.device)
                if probe_info["loss_name"] == "MSE"
                else t.nn.BCELoss(reduction="none").to(self.device)
            )

            self.probes[layer] = (classifier, alpha, loss_fn)

    def loss_fn(
        self, activations: Float[t.Tensor, "batch_size d_model"]
    ) -> Float[t.Tensor, "batch_size"]:
        loss = t.zeros(activations.shape[0], 1).to(self.device)

        for (classifier, alpha, lfn), layer in zip(
            self.probes.values(), self.config.layers
        ):
            act_name = tl.utils.get_act_name(self.config.pattern, layer)
            acts = self.act_dict[act_name][..., -1, :]

            prediction = classifier(acts).to(self.device)
            target = t.ones_like(prediction).to(self.device)

            loss += alpha * lfn(prediction, target)

        self.act_dict = {}

        return loss.squeeze(-1)
