import json
from typing import Dict, List, Literal, Optional, Tuple

import torch as t
import tqdm
import transformer_lens as tl
from jaxtyping import Float
from pydantic import PositiveInt

from ssr import PROBES_CONFIG_PATH, PROBES_WEIGTHS_PATH
from ssr.core import SSR, SSRConfig
from ssr.files import load_metadata, load_weights, save_metadata, save_weights
from ssr.lens import DEFAULT_VALUE, DefaultValue, Lens
from ssr.probes.classifiers import (
    LinearClassifier,
    activations_to_dataloader,
    train_and_test_classifier,
)
from ssr.types import HookList, Loss


class ProbeSSRConfig(SSRConfig):
    model_name: str
    layers: List[int]
    alphas: List[float]
    pattern: str = "resid_post"


class ProbeSSR(SSR):
    def __init__(
        self,
        lens: Lens,
        config: ProbeSSRConfig,
        load_probes: bool = True,
        probes_directory: str = str(PROBES_WEIGTHS_PATH),
    ):
        super().__init__(lens.model, config)

        self.lens = lens
        self.config: ProbeSSRConfig = config

        self.probes: Dict[int, Tuple[t.nn.Module, float, Loss]] = {}
        self.act_dict: Dict[str, t.Tensor] = {}
        self.fwd_hooks: HookList

        self.setup_hooks()
        if load_probes:
            self.load_probes(probes_directory)

    def setup_hooks(self):
        # Has to be called again if config.layers have changed
        if len(self.config.layers) > 0:
            self.fwd_hooks = [
                (tl.utils.get_act_name(self.config.pattern, layer), self.hook_fn)
                for layer in self.config.layers
            ]
            self.config.max_layer = max(self.config.layers)
        else:
            print("WARNING: layers is empty")

    def hook_fn(self, activations, hook):
        self.act_dict[hook.name] = activations.float()
        return activations

    def load_probes(self, probes_directory: str):
        best_configs = load_metadata(probes_directory, self.config.model_name)
        for layer, alpha in tqdm.tqdm(zip(self.config.layers, self.config.alphas)):
            probe_weights = load_weights(
                probes_directory,
                self.config.model_name,
                f"probe_layer_{layer}.pt",
            )

            classifier = LinearClassifier(self.model.cfg.d_model).to(self.device)
            classifier.load_state_dict(probe_weights)
            classifier.eval()

            for param in classifier.parameters():
                param.requires_grad = False

            loss_fn = (
                t.nn.MSELoss(reduction="none").to(self.device)
                if best_configs["loss_names"][layer] == "MSE"
                else t.nn.BCELoss(reduction="none").to(self.device)
            )

            self.probes[layer] = (classifier, alpha, loss_fn)

    def create_probes(
        self,
        verbose: bool = False,
        override_weights: bool = False,
        override_metadata: bool = False,
        layers: Optional[List[int]] = None,
        probes_directory: str = str(PROBES_WEIGTHS_PATH),
        probes_config_path: str = str(PROBES_CONFIG_PATH),
        pattern: DefaultValue | str = DEFAULT_VALUE,
        stack_act_name: DefaultValue | str = DEFAULT_VALUE,
        reduce_seq_method: DefaultValue
        | Literal["mean", "max", "last"] = DEFAULT_VALUE,
        system_message: DefaultValue | Optional[str] = DEFAULT_VALUE,
        dataset_name: DefaultValue
        | Literal["mod", "adv", "mini", "bomb"] = DEFAULT_VALUE,
        padding: DefaultValue | bool = DEFAULT_VALUE,
        padding_side: DefaultValue | Literal["left", "right"] = DEFAULT_VALUE,
        max_samples: DefaultValue | PositiveInt = DEFAULT_VALUE,
    ) -> None:
        hf_act, hl_act = self.lens.auto_scan_dataset(
            dataset_name=dataset_name,
            max_samples=max_samples,
            padding=padding,
            padding_side=padding_side,
            pattern=pattern,
            stack_act_name=stack_act_name,
            system_message=system_message,
            reduce_seq_method=reduce_seq_method,
        )
        probes_metrics = []

        with open(probes_config_path, "r") as file:
            probes_config = json.load(file)
        if layers is None:
            layers = list(range(self.lens.model.cfg.n_layers))

        probes_config_values = probes_config[self.config.model_name].values()
        loss_names = [x["loss_name"] for x in probes_config_values]
        optimizer_names = [x["optimizer"] for x in probes_config_values]
        lrs = [x["lr"] for x in probes_config_values]
        epochs = [x["epochs"] for x in probes_config_values]

        for layer, loss_name, optimizer, lr, epochs_ in tqdm.tqdm(
            zip(
                layers,
                loss_names,
                optimizer_names,
                lrs,
                epochs,
            )
        ):
            train_loader, test_loader, _ = activations_to_dataloader(
                hf_act[layer], hl_act[layer]
            )
            classifier, _, metrics = train_and_test_classifier(
                train_loader,
                test_loader,
                d_model=self.lens.model.cfg.d_model,
                loss_name=loss_name,
                optimizer_name=optimizer,
                lr=lr,
                epochs=epochs_,
            )
            classifier = classifier.to(self.device).float().eval()
            for param in classifier.parameters():
                param.requires_grad = False

            if verbose:
                print(f"Trained probe at layer: {layer}, with metrics: {metrics}.")

            probes_metrics.append(metrics)
            save_weights(
                probes_directory,
                self.config.model_name,
                f"probe_layer_{layer}.pt",
                classifier.state_dict(),
                override=override_weights,
            )

        save_metadata(
            probes_directory,
            self.config.model_name,
            self.config.model_dump()
            | {
                "loss_names": loss_names,
                "optimizer_names": optimizer_names,
                "lrs": lrs,
                "epochs": epochs,
                "metrics": probes_metrics,
            },
            override=override_metadata,
        )

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
