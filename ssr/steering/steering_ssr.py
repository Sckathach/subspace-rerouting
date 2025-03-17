from typing import Dict, List, Literal, Optional, Tuple

import einops
import torch as t
import transformer_lens as tl
from jaxtyping import Float
from pydantic import PositiveInt

from ssr import REFUSAL_DIRECTIONS_PATH
from ssr.core import SSR, SSRConfig
from ssr.defaults import DEFAULT_VALUE, DefaultValue
from ssr.files import load_weights, save_metadata, save_weights
from ssr.lens import Lens
from ssr.types import HookList


class SteeringSSRConfig(SSRConfig):
    model_name: str
    layers: List[int]
    alphas: List[float]
    pattern: str = "resid_post"


class SteeringSSR(SSR):
    def __init__(
        self,
        lens: Lens,
        config: SteeringSSRConfig,
        load_directions: bool = True,
        refusal_directions_path: str = str(REFUSAL_DIRECTIONS_PATH),
    ):
        super().__init__(lens.model, config)

        self.lens = lens
        self.config: SteeringSSRConfig = config

        self.refusal_directions: Float[t.Tensor, "n_layers d_model"]
        self.act_dict: Dict[str, t.Tensor] = {}
        self.fwd_hooks: HookList

        self.setup_hooks()
        if load_directions:
            self.load_refusal_directions(refusal_directions_path)

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
        self.act_dict[hook.name] = activations.cpu()
        return activations

    def load_refusal_directions(self, refusal_direction_path: str):
        self.refusal_directions = load_weights(
            refusal_direction_path,
            self.config.model_name,
            "refusal_directions.pt",
        ).to(self.device)

    def create_directions(
        self,
        override_metadata: bool = False,
        override_weights: bool = False,
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
        refusal_direction_path: str = str(REFUSAL_DIRECTIONS_PATH),
    ) -> None:
        hf_act, hl_act = self.lens.auto_scan_dataset(
            reduce_seq_method=reduce_seq_method,
            dataset_name=dataset_name,
            max_samples=max_samples,
            padding=padding,
            padding_side=padding_side,
            pattern=pattern,
            stack_act_name=stack_act_name,
            system_message=system_message,
        )

        refusal_directions = hf_act.mean(dim=1) - hl_act.mean(dim=1)
        refusal_directions = (
            refusal_directions
            / t.linalg.norm(refusal_directions, dim=-1, keepdim=True).cpu()
        )

        save_weights(
            directory=refusal_direction_path,
            model_name=self.config.model_name,
            weights_filename="refusal_directions.pt",
            weights=refusal_directions,
            override=override_weights,
        )

        save_metadata(
            directory=refusal_direction_path,
            model_name=self.config.model_name,
            metadata=self.config.model_dump(),
            override=override_metadata,
        )

    def decompose(
        self, activations: Float[t.Tensor, "... d_model"], layer: int
    ) -> Tuple[
        Float[t.Tensor, "... d_model"],
        Float[t.Tensor, "... d_model"],
        Float[t.Tensor, "... 1"],
    ]:
        proj_scal = einops.einsum(
            activations,
            self.refusal_directions[layer].view(-1, 1),
            "... d_model, d_model single -> ... single",
        )
        proj = proj_scal * self.refusal_directions[layer]

        proj_orth = activations - proj

        return proj, proj_orth, proj_scal

    def loss_fn(self, activations) -> Float[t.Tensor, "batch 1"]:
        loss = t.zeros(activations.shape[0]).cpu()

        for s, layer in zip(self.config.alphas, self.config.layers):
            act_name = tl.utils.get_act_name(self.config.pattern, layer)
            acts = self.act_dict[act_name][..., -1, :]

            proj = t.cosine_similarity(acts, self.refusal_directions[layer], dim=-1)

            loss += (-s - proj) ** 2

        self.act_dict = {}

        return loss.unsqueeze(-1)
