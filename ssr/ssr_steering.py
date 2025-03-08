from typing import Dict, List, Literal, Tuple

import einops
import torch as t
import transformer_lens as tl
from jaxtyping import Float
from pydantic import PositiveFloat, PositiveInt

from ssr import DEVICE
from ssr.core import SSR, SSRConfig
from ssr.datasets import get_max_seq_len, load_dataset, process_dataset, scan_dataset
from ssr.types import HookList


class SteeringSSRConfig(SSRConfig):
    loss_name: Literal["match", "slide"] = "match"
    layers: List[int] = [10, 15, 20]
    s_list: List[float] = [2.0]
    beta: PositiveFloat = 0.1

    dataset_name: Literal["adv", "mod"] = "mod"
    max_samples: PositiveInt = 124
    padding: bool = True
    padding_side: str = "left"

    pattern: str = "resid_post"
    stack_act_name: str = "resid_post"
    reduce_seq_method: Literal["mean", "max", "last"] = "last"


class SteeringSSR(SSR):
    def __init__(
        self,
        model: tl.HookedTransformer,
        config: SteeringSSRConfig,
        device: str = DEVICE,
    ):
        super().__init__(model, config, device)

        self.config: SteeringSSRConfig = config

        self.refusal_directions: Float[t.Tensor, "n_layers d_model"]
        self.act_dict: Dict[str, t.Tensor] = {}
        self.fwd_hooks: HookList

        self.setup_hooks()
        self.init_directions()

    def setup_hooks(self):
        # Has to be called again if config.layers have changed
        self.fwd_hooks = [
            (tl.utils.get_act_name(self.config.pattern, layer), self.hook_fn)
            for layer in self.config.layers
        ]
        self.config.max_layer = max(self.config.layers)

    def hook_fn(self, activations, hook):
        self.act_dict[hook.name] = activations.cpu()
        return activations

    def init_directions(self):
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
            max_samples=self.config.max_samples,
            padding_side=self.config.padding_side,
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

        refusal_directions = hf_act.mean(dim=1) - hl_act.mean(dim=1)
        self.refusal_directions = (
            refusal_directions
            / t.linalg.norm(refusal_directions, dim=-1, keepdim=True).cpu()
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

        for s, layer in zip(self.config.s_list, self.config.layers):
            act_name = tl.utils.get_act_name(self.config.pattern, layer)
            acts = self.act_dict[act_name][..., -1, :]

            proj = t.cosine_similarity(acts, self.refusal_directions[layer], dim=-1)

            loss += (-s - proj) ** 2

        self.act_dict = {}

        return loss.unsqueeze(-1)
