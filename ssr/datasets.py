import json
from typing import List, Literal, Optional, Tuple

import torch as t
from jaxtyping import Float, Int

from ssr.lens import Lens


def load_dataset(
    dataset_name: Literal["mod", "adv", "mini", "bomb"] = "mod",
    max_samples: int = 120,
) -> Tuple[List[str], List[str]]:
    match dataset_name:
        case "mod":
            with open("datasets/advbench_modified.json", "r") as f:
                dataset = json.load(f)
                hf_raw, hl_raw = (
                    [x["harmful"] for x in dataset],
                    [x["harmless"] for x in dataset],
                )

                return hf_raw[:max_samples], hl_raw[:max_samples]

        case "adv":
            with open("datasets/advbench_alpaca.json", "r") as f:
                dataset = json.load(f)
                hf_raw, hl_raw = dataset["harmful"], dataset["harmless"]

                return hf_raw[:max_samples], hl_raw[:max_samples]

        case "mini":
            with open("datasets/minibench.json", "r") as f:
                minibench = json.load(f)

                return minibench["harmful"][:max_samples], minibench["harmless"][
                    :max_samples
                ]

        case "bomb":
            with open("datasets/bomb_extended.json", "r") as f:
                bomb_extended = json.load(f)

                return bomb_extended["harmful"][:max_samples], bomb_extended[
                    "harmless"
                ][:max_samples]


def process_dataset(
    lens: Lens,
    hf_raw: List[str],
    hl_raw: List[str],
    seq_len: Optional[int] = None,
    max_samples: Optional[int] = None,
    padding_side: Optional[str] = None,
    system_message: Optional[str] = None,
) -> Tuple[Int[t.Tensor, "batch_size seq_len"], Int[t.Tensor, "batch_size seq_len"]]:
    max_samples = (
        max_samples if max_samples is not None else max(len(hf_raw), len(hl_raw))
    )

    match seq_len:
        case None:
            if padding_side is not None:
                lens.tokenizer.padding_side = padding_side

            if max_samples is not None:
                hf_raw = hf_raw[:max_samples]
                hl_raw = hl_raw[:max_samples]

            hf_ = [
                lens.apply_chat_template(p, system_message=system_message)
                for p in hf_raw
            ]
            hl_ = [
                lens.apply_chat_template(p, system_message=system_message)
                for p in hl_raw
            ]

            hf = lens.tokenizer(
                hf_, padding=True, return_tensors="pt", add_special_tokens=False
            ).input_ids
            hl = lens.tokenizer(
                hl_, padding=True, return_tensors="pt", add_special_tokens=False
            ).input_ids

            return hf, hl

        case _:
            hf_ = [
                tokens
                for p in hf_raw
                if len(
                    tokens := lens.apply_chat_template(
                        p, tokenize=True, system_message=system_message
                    )
                )
                == seq_len
            ]
            hl_ = [
                tokens
                for p in hl_raw
                if len(
                    tokens := lens.apply_chat_template(
                        p, tokenize=True, system_message=system_message
                    )
                )
                == seq_len
            ]

            min_len = min(len(hf_), len(hl_), max_samples)
            hf_ = hf_[:min_len]
            hl_ = hl_[:min_len]

            hf = t.cat([t.Tensor(p).unsqueeze(0).long() for p in hf_], dim=0)
            hl = t.cat([t.Tensor(p).unsqueeze(0).long() for p in hl_], dim=0)

            return hf, hl


def scan_dataset(
    lens: Lens,
    hf: Int[t.Tensor, "batch_size seq_len"],
    hl: Int[t.Tensor, "batch_size seq_len"],
    pattern: str = "resid_mid",
    stack_act_name: str = "resid_mid",
    reduce_seq_method: Literal["last", "mean", "max"] = "last",
) -> Tuple[
    Float[t.Tensor, "n_layers batch_size d_model"],
    Float[t.Tensor, "n_layers batch_size d_model"],
]:
    hf_scan = lens.auto_scan(hf, pattern=pattern)
    hl_scan = lens.auto_scan(hl, pattern=pattern)

    try:
        hf_act = hf_scan.stack_activation(stack_act_name)
        hl_act = hl_scan.stack_activation(stack_act_name)
    except Exception as e:
        raise ValueError(
            f"Cannot stack activations! Check stack_act_name. Error: {e}"
        ) from e

    match reduce_seq_method:
        case "last":
            return hf_act[:, :, -1, :], hl_act[:, :, -1, :]
        case "mean":
            return hf_act.mean(dim=2), hl_act.mean(dim=2)
        case "max":
            return hf_act.max(dim=2)[0], hl_act.max(dim=2)[0]
        case _:
            raise ValueError(f"reduce_seq_method {reduce_seq_method} doesn't exist")


def auto_scan_dataset(
    lens: Lens,
    dataset_name: Literal["adv", "mod"] = "mod",
    max_samples: int = 32,
    pattern: str = "resid_post",
    stack_act_name: str = "resid_post",
    reduce_seq_method: Literal["mean", "max", "last"] = "last",
) -> Tuple[
    Float[t.Tensor, "n_layers batch_size d_model"],
    Float[t.Tensor, "n_layers batch_size d_model"],
]:
    hf_raw, hl_raw = load_dataset(dataset_name=dataset_name)
    seq_len = get_max_seq_len(lens, hf_raw, hl_raw)
    hf, hl = process_dataset(
        lens, hf_raw, hl_raw, seq_len=seq_len[0], max_samples=max_samples
    )
    return scan_dataset(
        lens,
        hf,
        hl,
        pattern=pattern,
        stack_act_name=stack_act_name,
        reduce_seq_method=reduce_seq_method,
    )


def get_max_seq_len(
    lens: Lens, a_raw: List[str], b_raw: List[str], **kwargs
) -> Tuple[int, int]:
    """
    Returns:
        Tuple[int, int]: seq_len, n_samples
    """
    from collections import Counter

    a_tokens = [lens.apply_chat_template(x, tokenize=True, **kwargs) for x in a_raw]
    b_tokens = [lens.apply_chat_template(x, tokenize=True, **kwargs) for x in b_raw]

    a_seq_lens = [len(x) for x in a_tokens]
    b_seq_lens = [len(x) for x in b_tokens]

    a_dict, b_dict = (
        dict(Counter(a_seq_lens).most_common()),
        dict(Counter(b_seq_lens).most_common()),
    )
    s = [(k, min(v, b_dict.get(k, 0))) for k, v in a_dict.items()]

    # Should already be sorted as list(dict()) sort the elements of the dict by their (execution time) insertion order
    s.sort(key=lambda x: x[1], reverse=True)

    return s[0]
