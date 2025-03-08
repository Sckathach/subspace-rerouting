import gc
from typing import List, Optional, cast

import torch as t
import tqdm
import transformer_lens as tl
from jaxtyping import Float, Int
from pydantic import BaseModel, NonNegativeInt, PositiveFloat, PositiveInt
from rich.markup import escape

from ssr import DEBUG_SSR_DEFAULT, DEVICE, pprint
from ssr.gcg_utils import filter_ids, get_restricted_tokens, sample_ids_from_grad
from ssr.lens import Lens
from ssr.memory import find_executable_batch_size
from ssr.types import HookList


class SSRConfig(BaseModel):
    total_iterations: PositiveInt = 250
    suffix_length: PositiveInt = 15
    search_width: PositiveInt = 1024
    max_layer: NonNegativeInt = 0  # ie: >= 0..;
    search_topk: PositiveInt = 256
    replace_coefficient: PositiveFloat = 1.8
    allow_non_ascii: bool = False
    early_stop_loss: float = 0.25
    system_message: Optional[str] = "You are a helpful assistant."
    buffer_size: PositiveInt = 162
    patience: PositiveInt = 30


class SSR(Lens):
    def __init__(
        self, model: tl.HookedTransformer, config: SSRConfig, device: str = DEVICE
    ):
        super().__init__(model)

        self.fixed_n_replace: Optional[int] = None
        self.loss_history: List[float] = []

        from rich.traceback import install

        install(show_locals=DEBUG_SSR_DEFAULT)

        self.config = config

        self.not_allowed_ids = get_restricted_tokens(
            self.tokenizer,
            allow_non_ascii=self.config.allow_non_ascii,
            model_name=self.model.cfg.model_name,
        )

        self.initial_loss: float
        self.fwd_hooks: HookList
        self.n_replace: int
        self.device: str = device

        self.before_embeds: Float[t.Tensor, "seq_before d_model"]
        self.after_embeds: Float[t.Tensor, "seq_after d_model"]

        self.candidates_ids: Int[t.Tensor, "buffer_size adv_len"]
        self.candidates_losses: Float[t.Tensor, "buffer_size"]
        self.archive_candidates_ids: Int[t.Tensor, "buffer_size adv_len"]
        self.archive_candidates_losses: Float[t.Tensor, "buffer_size"]

    def loss_fn(
        self, activations: Float[t.Tensor, "... d_model"]
    ) -> Float[t.Tensor, "... 1"]:
        raise NotImplementedError

    def init_prompt(self, prompt: str):
        # TODO: adv anywhere not only suffix
        prompt = self.apply_chat_template(
            f"{prompt}[BOP]",
            system_message=self.config.system_message,
        )
        before_prompt, after_prompt = prompt.split("[BOP]")
        self.before_embeds = t.embedding(
            self.model.W_E, self.model.to_tokens(before_prompt, prepend_bos=False)
        ).squeeze(0)
        self.after_embeds = t.embedding(
            self.model.W_E, self.model.to_tokens(after_prompt, prepend_bos=False)
        ).squeeze(0)

    def compute_gradients(
        self, adversarial_one_hot: Int[t.Tensor, "adv_len d_vocab"]
    ) -> Float[t.Tensor, "adv_len d_vocab"]:
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        adversarial_embeds = adversarial_one_hot @ self.model.W_E
        full_embeds = t.cat(
            [self.before_embeds, adversarial_embeds, self.after_embeds], dim=0
        )

        with t.set_grad_enabled(True) and self.model.hooks(
            self.fwd_hooks
        ) as hooked_model:
            output = hooked_model.forward(  # type: ignore
                full_embeds.unsqueeze(0),
                start_at_layer=0,
                stop_at_layer=self.config.max_layer + 1,
                prepend_bos=False,
            )  # type: ignore

            loss = self.loss_fn(output)
            loss.backward(retain_graph=True)

            del output
            gc.collect()
            t.cuda.empty_cache()

        return adversarial_one_hot.grad.detach()  # type: ignore

    @t.no_grad()
    def compute_candidates_loss(
        self,
        search_batch_size: int,
        input_embeds: Float[t.Tensor, "search_width seq_len d_model"],
    ) -> Float[t.Tensor, "search_width"]:
        all_loss = []

        with self.model.hooks(self.fwd_hooks) as hooked_model:
            for i in range(0, input_embeds.shape[0], search_batch_size):
                input_embeds_batch = input_embeds[i : i + search_batch_size].to(
                    self.device
                )

                output = hooked_model.forward(  # type: ignore
                    input_embeds_batch,
                    start_at_layer=0,
                    stop_at_layer=self.config.max_layer + 1,
                )  # type: ignore

                loss = self.loss_fn(output)
                all_loss.append(loss)

                del output, input_embeds_batch
                gc.collect()
                t.cuda.empty_cache()

        return t.cat(all_loss, dim=0).squeeze(1)

    def update_n_replace(self):
        if self.fixed_n_replace is not None:
            self.n_replace = self.fixed_n_replace
        else:
            loss_ratio = min(
                1,
                max(0, self.candidates_losses[0].item() / (self.initial_loss + 1e-5)),
            ) ** (1 / self.config.replace_coefficient)
            new_n_replace = max(1, int(loss_ratio * self.config.suffix_length))

            pprint(f"Decreasing n_replace from {self.n_replace} to {new_n_replace}")
            self.n_replace = new_n_replace

    def update_candidates_buffer(
        self,
        candidates_ids: Int[t.Tensor, "search_width adv_len"],
        losses: Float[t.Tensor, "search_width"],
    ):
        pool = t.cat([self.candidates_losses, losses.cpu()], dim=-1)
        ordered_pool = t.topk(pool, k=pool.shape[0], largest=False)

        duplicate_mask = t.cat(
            [
                t.Tensor([1]).long(),
                (1 - (ordered_pool.values[:-1] == ordered_pool.values[1:]).long()),
            ],
            dim=-1,
        ).bool()

        filtered_pool_ids = ordered_pool.indices[duplicate_mask][
            : self.config.buffer_size
        ]
        self.candidates_losses = ordered_pool.values[duplicate_mask][
            : self.config.buffer_size
        ]

        self.candidates_ids = t.cat([self.candidates_ids, candidates_ids], dim=0)[
            filtered_pool_ids
        ]

    def archive_candidate(self):
        jump_idx = t.multinomial(t.softmax(-self.candidates_losses, dim=-1), 1)
        pprint(
            f"[b][red]Patience max reached, jumping from {self.candidates_ids[0]} with {self.candidates_losses[0]} to {self.candidates_ids[jump_idx]} with {self.candidates_losses[jump_idx]} ({jump_idx.item()} jump)[/][/]"
        )
        # pprint(self.candidates_ids, self.candidates_losses)
        self.archive_candidates_ids = t.cat(
            [
                self.archive_candidates_ids,
                self.candidates_ids[:jump_idx].cpu(),
            ],
            dim=0,
        )
        self.archive_candidates_losses = t.cat(
            [
                self.archive_candidates_losses,
                self.candidates_losses[:jump_idx].cpu(),
            ],
            dim=0,
        )
        self.candidates_ids = self.candidates_ids[jump_idx:]
        self.candidates_losses = self.candidates_losses[jump_idx:]

    def init_buffers(self):
        candidates_ids = (
            t.randint(
                0,
                self.model.cfg.d_vocab,
                (self.config.buffer_size, self.config.suffix_length),
            )
            .long()
            .to(self.device)
        )
        full_embeds = t.cat(
            [
                self.before_embeds.unsqueeze(0).repeat(self.config.buffer_size, 1, 1),
                t.embedding(self.model.W_E, candidates_ids),
                self.after_embeds.unsqueeze(0).repeat(self.config.buffer_size, 1, 1),
            ],
            dim=1,
        )
        candidates_losses = cast(
            Float[t.Tensor, "buffer_size 1"],
            find_executable_batch_size(
                self.compute_candidates_loss, self.config.buffer_size
            )(full_embeds),
        ).squeeze(-1)

        self.candidates_ids = t.empty(0, dtype=candidates_ids.dtype).to(self.device)
        self.candidates_losses = t.empty(0, dtype=candidates_losses.dtype)
        self.archive_candidates_ids = t.empty(0, dtype=candidates_ids.dtype)
        self.archive_candidates_losses = t.empty(0, dtype=candidates_losses.dtype)

        self.update_candidates_buffer(candidates_ids, candidates_losses)
        self.initial_loss = self.candidates_losses[0].item()

    def generate(self) -> None:
        last_update = 0

        for i in tqdm.tqdm(range(self.config.total_iterations)):
            optim_ids = self.candidates_ids[0].clone()

            if i == 0:
                if self.fixed_n_replace is None:
                    self.n_replace = len(optim_ids)
                else:
                    self.n_replace = self.fixed_n_replace

            optim_ids_one_hot = (
                t.nn.functional.one_hot(optim_ids, self.model.cfg.d_vocab)
                .half()
                .detach()
            )
            optim_ids_one_hot.requires_grad = True

            grad = self.compute_gradients(optim_ids_one_hot)

            with t.no_grad():
                candidate_ids = sample_ids_from_grad(
                    optim_ids,
                    grad,
                    search_width=self.config.search_width,
                    topk=self.config.search_topk,
                    n_replace=self.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )  # new_search_width adv_len d_model

                candidate_ids = filter_ids(candidate_ids, self.tokenizer)

                new_search_width = candidate_ids.shape[0]

                full_embeds = t.cat(
                    [
                        self.before_embeds.unsqueeze(0).repeat(new_search_width, 1, 1),
                        t.embedding(
                            self.model.W_E, candidate_ids.long().to(self.device)
                        ),
                        self.after_embeds.unsqueeze(0).repeat(new_search_width, 1, 1),
                    ],
                    dim=1,
                )

                losses = cast(
                    Float[t.Tensor, "new_search_width 1"],
                    find_executable_batch_size(
                        self.compute_candidates_loss, new_search_width
                    )(full_embeds),
                ).squeeze(-1)

                min_loss = losses.min().item()  # type: ignore
                self.loss_history.append(min_loss)

                if min_loss < self.candidates_losses[0].item():
                    self.update_n_replace()
                    last_update = 0

                    pprint(
                        f"[b][yellow]Best loss: {self.candidates_losses[0].item():.3f}, with ids: {self.candidates_ids[0]}\nTesting: {escape(cast(str, self.model.to_string(optim_ids)))}[/][/]"
                    )
                else:
                    last_update += 1

                self.update_candidates_buffer(candidate_ids, losses)

                if self.candidates_losses[0] < self.config.early_stop_loss:
                    pprint("[b][green]Early loss achived, exiting.[/][/]")
                    break

                if last_update > self.config.patience:
                    if len(self.candidates_ids) > 1:
                        self.archive_candidate()
                        last_update = 0
                    else:
                        pprint("[b][red]Candidates exhausted, exiting..., bye![/][/]")
                        break

                del losses
                t.cuda.empty_cache()
                gc.collect()
