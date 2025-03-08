from typing import Dict, List, Optional

import torch as t
import transformer_lens as tl
from pydantic import BaseModel

from ssr import DEVICE
from ssr.core import SSR, SSRConfig
from ssr.types import HookList


class Dazzle(BaseModel):
    layer: int
    head_index: int
    target_slice_start: int = -1
    target_slice_end: Optional[int] = None


class Ablation(BaseModel):
    layer: int
    head_index: int
    source_position: int
    target_position: int


class FullAblation(BaseModel):
    layer: int
    head_index: int
    target_position: int


class DazzleScore(BaseModel):
    layer: int
    head_index: int
    target_slice_start: int = -1
    target_slice_end: Optional[int] = None


class AttentionSSRConfig(SSRConfig):
    interventions: List[Dazzle | Ablation | FullAblation | DazzleScore]


class AttentionSSR(SSR):
    def __init__(
        self,
        model: tl.HookedTransformer,
        config: AttentionSSRConfig,
        device: str = DEVICE,
    ):
        super().__init__(model, config, device)

        self.config: AttentionSSRConfig = config

        self.act_dict: Dict[str, t.Tensor] = {}
        self.fwd_hooks: HookList = []

        self.setup_hooks()

    def setup_hooks(self):
        for intervention in self.config.interventions:
            if isinstance(intervention, Dazzle | Ablation):
                self.fwd_hooks.append(
                    (tl.utils.get_act_name("pattern", intervention.layer), self.hook_fn)
                )
            elif isinstance(intervention, DazzleScore):
                self.fwd_hooks.append(
                    (
                        tl.utils.get_act_name("attn_scores", intervention.layer),
                        self.hook_fn,
                    )
                )
            else:
                self.fwd_hooks.append(
                    (tl.utils.get_act_name("z", intervention.layer), self.hook_fn)
                )

        self.config.max_layer = max(
            [intervention.layer for intervention in self.config.interventions]
        )

    def hook_fn(self, activations, hook):
        self.act_dict[hook.name] = activations.cpu()
        return activations

    def loss_fn(self, activations):
        loss = t.zeros(activations.shape[0], 1)

        for intervention in self.config.interventions:
            if isinstance(intervention, Dazzle):
                if intervention.target_slice_end is None:
                    rect_height = -intervention.target_slice_start
                else:
                    rect_height = (
                        intervention.target_slice_end - intervention.target_slice_start
                    )

                act_name = tl.utils.get_act_name("pattern", intervention.layer)
                pattern = self.act_dict[act_name]
                loss += rect_height - (
                    pattern[
                        :,
                        intervention.head_index,
                        intervention.target_slice_start : intervention.target_slice_end,
                        self.before_embeds.shape[0] : -self.after_embeds.shape[0],
                    ]
                    .sum(-1)
                    .sum(-1)
                    .unsqueeze(-1)
                )

            elif isinstance(intervention, Ablation):
                act_name = tl.utils.get_act_name("pattern", intervention.layer)
                pattern = self.act_dict[act_name]
                loss += pattern[
                    :,
                    intervention.head_index,
                    intervention.source_position,
                    intervention.target_position,
                ].unsqueeze(-1)

            elif isinstance(intervention, FullAblation):
                act_name = tl.utils.get_act_name("z", intervention.layer)
                hook_z = self.act_dict[act_name]
                loss += t.linalg.norm(
                    hook_z[
                        :, intervention.target_position :, intervention.head_index, :
                    ],
                    dim=-1,
                    ord=2,
                ).unsqueeze(-1)

            else:
                act_name = tl.utils.get_act_name("attn_scores", intervention.layer)
                scores = self.act_dict[act_name]
                loss -= (
                    scores[
                        :,
                        intervention.head_index,
                        intervention.target_slice_start : intervention.target_slice_end,
                        self.before_embeds.shape[0] : -self.after_embeds.shape[0],
                    ]
                    .sum(-1)
                    .sum(-1)
                    .unsqueeze(-1)
                )

        self.act_dict = {}

        if activations.shape[0] == 1:
            return loss[0]

        return loss.unsqueeze(-1)
