from typing import List, Optional, cast

import torch as t
from torch.types import Device

from ssr.lens import get_surname, model_info
from ssr.types import Tokenizer


def get_restricted_tokens(
    tokenizer: Tokenizer,
    device: Optional[Device] = "cpu",
    allow_non_ascii: bool = False,
    model_name: Optional[str] = None,
    restricted_tokens_list: Optional[List[str | int]] = None,
) -> t.Tensor:
    restricted_tokens = t.tensor(tokenizer.all_special_ids)

    if (
        model_name is not None
        and (
            restricted_tokens_list := cast(
                Optional[List[str | int]],
                model_info(get_surname(model_name)).get("restricted_tokens"),
            )
        )
    ) or restricted_tokens_list is not None:
        for token in restricted_tokens_list:
            if isinstance(token, str):
                restricted_tokens = t.cat(
                    [
                        restricted_tokens,
                        t.arange(int(token.split("-")[0]), int(token.split("-")[1])),
                    ]
                )
            else:
                restricted_tokens = t.cat([restricted_tokens, t.Tensor(token)])

    if not allow_non_ascii:
        decoded_vocabulary = tokenizer.batch_decode(list(range(tokenizer.vocab_size)))

        def is_ascii(s: str) -> bool:
            return s.isascii() and s.isprintable()

        non_ascii_toks = []

        for i, v in enumerate(decoded_vocabulary):
            if not is_ascii(v):
                non_ascii_toks.append(i)

        restricted_tokens = t.cat([restricted_tokens, t.tensor(non_ascii_toks)], dim=0)

    return restricted_tokens.to(device).long()


# Code adapted from GraySwanAI's nanoGCG: https://github.com/GraySwanAI/nanoGCG
def sample_ids_from_grad(
    ids: t.Tensor,
    grad: t.Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Optional[t.Tensor] = None,
):
    n_optim_tokens = len(ids)

    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices
    # print((-grad).topk(topk, dim=1))

    sampled_ids_pos = t.argsort(
        t.rand((search_width, n_optim_tokens), device=grad.device)
    )[..., :n_replace]

    sampled_ids_val = t.gather(
        topk_ids[sampled_ids_pos],
        2,
        t.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


# Code adapted from GraySwanAI's nanoGCG: https://github.com/GraySwanAI/nanoGCG
def filter_ids(ids: t.Tensor, tokenizer: Tokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
            token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = (
            tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False)
            .to(ids.device)
            .input_ids[0]
        )
        if t.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])

    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return t.stack(filtered_ids)
