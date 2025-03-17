import re
from typing import List, Optional, cast

import torch as t
import tqdm
import transformer_lens as tl
from accelerate.utils.memory import find_executable_batch_size, release_memory
from jaxtyping import Float, Int
from pydantic import BaseModel
from rich import print

from ssr.types import HookList, Tokenizer


class SSRConfig(BaseModel):
    model_name: str
    search_width: int = 256
    search_topk: int = 64
    buffer_size: int = 10
    replace_coefficient: float = 1.8
    n_replace: Optional[int] = None
    max_layer: int = -1
    patience: int = 10
    early_stop_loss: float = 0.05
    max_iterations: int = 60


class SSR:
    def __init__(self, model: tl.HookedTransformer, config: SSRConfig) -> None:
        self.model = model
        self.device = self.model.cfg.device

        self.not_allowed_ids = None

        if self.model.tokenizer is None:
            raise ValueError("model.tokenizer is supposed not None")
        self.tokenizer = cast(Tokenizer, self.model.tokenizer)

        self.config = config
        if self.config.max_layer < 0:
            self.config.max_layer += self.model.cfg.n_layers

        self.full_tokens: Int[t.Tensor, "seq_len"]
        self.full_embeds: Float[t.Tensor, "seq_len d_model"]
        self.mask_positions: Int[t.Tensor, "mask_len"]

        # Buffers
        self.candidate_ids: Int[t.Tensor, "buffer_size mask_len"]
        self.candidate_losses: Float[t.Tensor, "buffer_size"]
        self.archive_ids: Int[t.Tensor, "archive_size mask_len"]
        self.archive_losses: Float[t.Tensor, "archive_size"]

        self.initial_loss: float
        self.n_replace: int

        self.fwd_hooks: HookList

    def init_prompt(self, sentence: str, mask_str: str = "[MASK]") -> None:
        parts = re.split(f"({re.escape(mask_str)})", sentence)

        fixed_tokens = []
        fixed_positions: List[int] = []
        mask_positions = []
        current_pos = 0

        for part in parts:
            if part == mask_str:
                mask_positions.append(current_pos)
                current_pos += 1
            elif len(part) > 0:
                tokens = self.model.to_tokens(part, prepend_bos=False).squeeze(0)
                fixed_tokens.append(tokens)
                fixed_positions.extend(range(current_pos, current_pos + len(tokens)))
                current_pos += len(tokens)

        self.full_tokens = t.zeros(current_pos).to(self.model.cfg.device).long()
        self.full_tokens[fixed_positions] = (
            t.cat(fixed_tokens, dim=0).to(self.model.cfg.device).long()
        )
        self.full_embeds = t.embedding(self.model.W_E, self.full_tokens)
        self.mask_positions = t.Tensor(mask_positions).to(self.model.cfg.device).long()

    def get_tokens(
        self, masked_tokens: Int[t.Tensor, "batch_size mask_len"]
    ) -> Int[t.Tensor, "batch_size seq_len"]:
        full_tokens = (
            self.full_tokens.clone()
            .to(masked_tokens.device)
            .unsqueeze(0)
            .repeat(masked_tokens.shape[0], 1)
        )
        full_tokens[:, self.mask_positions] = masked_tokens
        return full_tokens

    def buffer_init_random(self):
        # TODO: filter ids
        candidate_ids = (
            t.randint(
                0,
                self.model.cfg.d_vocab,
                (self.config.buffer_size, len(self.mask_positions)),
            )
            .long()
            .to(self.full_embeds.device)
        )
        candidate_full_embeds = (
            self.full_embeds.clone().unsqueeze(0).repeat(self.config.buffer_size, 1, 1)
        )
        candidate_full_embeds[:, self.mask_positions, :] = t.embedding(
            self.model.W_E, candidate_ids
        )
        candidate_losses = cast(
            Float[t.Tensor, "buffer_size 1"],
            find_executable_batch_size(
                self.compute_candidate_losses, self.config.buffer_size
            )(candidate_full_embeds),
        ).squeeze(-1)
        self.candidate_ids = t.empty(0, dtype=candidate_ids.dtype).to(self.device)
        self.candidate_losses = t.empty(0, dtype=candidate_losses.dtype)
        self.archive_ids = t.empty(0, dtype=candidate_ids.dtype)
        self.archive_losses = t.empty(0, dtype=candidate_losses.dtype)
        self.buffer_add(candidate_ids, candidate_losses, update_n_replace=False)
        self.initial_loss = self.candidate_losses[0].item()
        self.n_replace = len(self.mask_positions)

    def buffer_add(
        self,
        candidates_ids: Int[t.Tensor, "search_width adv_len"],
        losses: Float[t.Tensor, "search_width"],
        update_n_replace: bool = True,
    ):
        best_loss_before = (
            self.candidate_losses[0].item() if len(self.candidate_losses) > 0 else 666.0
        )

        pool = t.cat([self.candidate_losses, losses.cpu()], dim=-1)
        ordered_pool = t.topk(pool, k=pool.shape[0], largest=False, dim=0)

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
        self.candidate_losses = ordered_pool.values[duplicate_mask][
            : self.config.buffer_size
        ]

        self.candidate_ids = t.cat([self.candidate_ids, candidates_ids], dim=0)[
            filtered_pool_ids
        ]
        if self.candidate_losses[0] < best_loss_before:
            print(f""" 
                [b][yellow]Best loss: {self.candidate_losses[0]:.3f}, with ids: {self.candidate_ids[0]}
            """)
        if update_n_replace:
            self.update_n_replace()

    def buffer_jump(self):
        jump_idx = t.multinomial(t.softmax(-self.candidate_losses, dim=-1), 1)

        print(
            f"[b][red]Patience max reached, jumping from {self.candidate_ids[0]} with "
            f"{self.candidate_losses[0].item()} to {self.candidate_ids[jump_idx]} with "
            f"{self.candidate_losses[jump_idx].item()} ({jump_idx.item()} jumps)[/][/]"
        )
        self.archive_ids = t.cat(
            [
                self.archive_ids,
                self.candidate_ids[:jump_idx].cpu(),
            ],
            dim=0,
        )
        self.archive_losses = t.cat(
            [
                self.archive_losses,
                self.candidate_losses[:jump_idx].cpu(),
            ],
            dim=0,
        )
        self.candidate_ids = self.candidate_ids[jump_idx:]
        self.candidate_losses = self.candidate_losses[jump_idx:]

    def update_n_replace(self):
        loss_ratio = min(
            1,
            max(0, self.candidate_losses[0].item() / (self.initial_loss + 1e-5)),
        ) ** (1 / self.config.replace_coefficient)
        new_n_replace = max(1, int(loss_ratio * len(self.mask_positions)))

        if new_n_replace != self.n_replace:
            print(f"Decreasing n_replace from {self.n_replace} to {new_n_replace}")
            self.n_replace = new_n_replace

    def sample_ids_from_grad(
        self, ids: Int[t.Tensor, "mask_len"], grad: Float[t.Tensor, "mask_len"]
    ):
        original_ids = ids.repeat(self.config.search_width, 1)

        if self.not_allowed_ids is not None:
            grad[:, self.not_allowed_ids.to(grad.device)] = float("inf")

        topk_ids = (-grad).topk(self.config.search_topk, dim=1).indices

        sampled_ids_pos = t.argsort(
            t.rand(
                (self.config.search_width, len(self.mask_positions)), device=grad.device
            )
        )[..., : self.n_replace]

        sampled_ids_val = t.gather(
            topk_ids[sampled_ids_pos],
            2,
            t.randint(
                0,
                self.config.search_topk,
                (self.config.search_width, self.n_replace, 1),
                device=grad.device,
            ),
        ).squeeze(2)

        new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

        return new_ids

    def loss_fn(
        self, activations: Float[t.Tensor, "batch_size ..."]
    ) -> Float[t.Tensor, "batch_size"]:
        raise NotImplementedError

    def compute_gradients(
        self, adv_one_hot: Int[t.Tensor, "adv_len d_vocab"]
    ) -> Float[t.Tensor, "adv_len d_vocab"]:
        self.model.eval()

        adv_embed = adv_one_hot @ self.model.W_E
        self.full_embeds[self.mask_positions] = adv_embed

        with t.set_grad_enabled(True) and self.model.hooks(
            self.fwd_hooks
        ) as hooked_model:
            output = hooked_model.forward(  # type: ignore
                self.full_embeds.unsqueeze(0),
                start_at_layer=0,
                stop_at_layer=self.config.max_layer + 1,
                prepend_bos=False,
            )  # type: ignore

            loss = self.loss_fn(output)
            loss.backward(retain_graph=True)

            output = release_memory(output)

        return adv_one_hot.grad.detach()  # type: ignore

    @t.no_grad()
    def compute_candidate_losses(
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

                output, input_embeds_batch = release_memory(output, input_embeds_batch)

        return t.cat(all_loss, dim=0)

    def generate(self) -> None:
        last_update = 0
        for i in tqdm.tqdm(range(self.config.max_iterations)):
            optim_ids = self.candidate_ids[0].clone()
            optim_ids_one_hot = (
                t.nn.functional.one_hot(optim_ids, self.model.cfg.d_vocab)
                .to(self.model.W_E.dtype)
                .detach()
            )
            optim_ids_one_hot.requires_grad = True

            grad = self.compute_gradients(optim_ids_one_hot)

            with t.no_grad():
                candidate_ids = self.sample_ids_from_grad(optim_ids, grad)

                candidate_ids = filter_tokens(self.tokenizer, candidate_ids)

                candidate_full_embeds = (
                    self.full_embeds.clone()
                    .unsqueeze(0)
                    .repeat(candidate_ids.shape[0], 1, 1)
                )
                candidate_full_embeds[:, self.mask_positions, :] = t.embedding(
                    self.model.W_E, candidate_ids
                )
                candidate_losses = cast(
                    Float[t.Tensor, "new_search_width"],
                    find_executable_batch_size(
                        self.compute_candidate_losses, candidate_full_embeds.shape[0]
                    )(candidate_full_embeds),
                )

                self.buffer_add(candidate_ids, candidate_losses)

                losses, candidate_embeds, candidate_full_ids = release_memory(
                    candidate_losses, candidate_full_embeds, candidate_ids
                )

                if i - last_update > self.config.patience:
                    last_update = i
                    self.buffer_jump()

                if self.candidate_losses[0] < self.config.early_stop_loss:
                    break


def get_restricted_tokens(
    tokenizer: Tokenizer, restricted_tokens_list: List[str | int]
) -> Int[t.Tensor, "n_restricted_tokens"]:
    restricted_tokens = t.tensor(tokenizer.all_special_ids)
    for token in restricted_tokens_list:
        if isinstance(token, str):
            if token == "non-ascii":
                decoded_vocabulary = tokenizer.batch_decode(
                    list(range(tokenizer.vocab_size))
                )

                def is_ascii(s: str) -> bool:
                    return s.isascii() and s.isprintable()

                non_ascii_toks = []

                for i, v in enumerate(decoded_vocabulary):
                    if not is_ascii(v):
                        non_ascii_toks.append(i)

                restricted_tokens = t.cat(
                    [restricted_tokens, t.tensor(non_ascii_toks)], dim=0
                )
            else:
                restricted_tokens = t.cat(
                    [
                        restricted_tokens,
                        t.arange(int(token.split("-")[0]), int(token.split("-")[1])),
                    ]
                )
        else:
            restricted_tokens = t.cat([restricted_tokens, t.Tensor(token)])

    return restricted_tokens.long()


def filter_tokens(
    tokenizer: Tokenizer, tokens: Int[t.Tensor, "batch seq_len"]
) -> Int[t.Tensor, "new_batch seq_len"]:
    str_tokens = tokenizer.batch_decode(tokens)
    filtered_tokens = []

    for i in range(len(str_tokens)):
        # Retokenize the decoded token ids
        ids_encoded = (
            tokenizer(str_tokens[i], return_tensors="pt", add_special_tokens=False)
            .to(tokens.device)
            .input_ids[0]
        )
        if t.equal(tokens[i], ids_encoded):
            filtered_tokens.append(tokens[i])

    if not filtered_tokens:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return t.stack(filtered_tokens)
