import json
from typing import Dict, List, Literal, Tuple

import torch as t
import tqdm
import transformer_lens as tl
from pydantic import PositiveInt
from rich import print

from ssr import DEVICE, PROBES_CONFIG_PATH
from ssr.classifiers import activations_to_dataloader, train_and_test_classifier
from ssr.core import SSR, SSRConfig
from ssr.datasets import get_max_seq_len, load_dataset, process_dataset, scan_dataset
from ssr.types import HookList, Loss


class ProbeSSRConfig(SSRConfig):
    model_name: str
    layers: List[int]
    alphas: List[float]

    dataset_name: Literal["adv", "mod"] = "mod"
    max_samples: PositiveInt = 124
    padding: bool = True
    padding_side: str = "left"

    pattern: str = "resid_post"
    stack_act_name: str = "resid_post"
    reduce_seq_method: Literal["mean", "max", "last"] = "last"


class ProbeSSR(SSR):
    def __init__(
        self, model: tl.HookedTransformer, config: ProbeSSRConfig, device: str = DEVICE
    ):
        super().__init__(model, config, device)

        self.config: ProbeSSRConfig = config

        self.probes: Dict[int, Tuple[t.nn.Module, float, Loss]] = {}
        self.act_dict: Dict[str, t.Tensor] = {}
        self.fwd_hooks: HookList

        self.setup_hooks()
        self.init_probes()

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

    def init_probes(self, verbose: bool = False):
        with open(PROBES_CONFIG_PATH, "r") as f:
            best_configs = json.load(f)[self.config.model_name]

        hf_raw, hl_raw = load_dataset(self.config.dataset_name)

        seq_len = (
            get_max_seq_len(self, hf_raw, hl_raw)[0]
            if not self.config.padding
            else None
        )

        hf, hl = process_dataset(
            self,
            hf_raw,
            hl_raw,
            padding_side=self.config.padding_side,
            max_samples=self.config.max_samples,
            system_message=self.config.system_message,
            seq_len=seq_len,
        )

        hf_act, hl_act = scan_dataset(
            self,
            hf,
            hl,
            pattern=self.config.pattern,
            stack_act_name=self.config.stack_act_name,
            reduce_seq_method=self.config.reduce_seq_method,
        )

        for layer, alpha in tqdm.tqdm(zip(self.config.layers, self.config.alphas)):
            train_loader, test_loader, _ = activations_to_dataloader(
                hf_act[layer], hl_act[layer]
            )
            classifier, _, metrics = train_and_test_classifier(
                train_loader,
                test_loader,
                d_model=self.model.cfg.d_model,
                loss_name=best_configs[str(layer)]["loss_name"],
                optimizer_name=best_configs[str(layer)]["optimizer"],
                lr=best_configs[str(layer)]["lr"],
                epochs=best_configs[str(layer)]["epochs"],
            )
            classifier = classifier.to(self.device).float().eval()
            for param in classifier.parameters():
                param.requires_grad = False

            if verbose:
                print(f"Trained probe at layer: {layer}, with metrics: {metrics}.")

            loss_fn = (
                t.nn.MSELoss(reduction="none").to(self.device)
                if best_configs[str(layer)]["loss_name"] == "MSE"
                else t.nn.BCELoss(reduction="none").to(self.device)
            )

            self.probes[layer] = (classifier, alpha, loss_fn)

    def loss_fn(self, activations):
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

        if activations.shape[0] == 1:
            return loss[0]

        return loss.unsqueeze(-1)
