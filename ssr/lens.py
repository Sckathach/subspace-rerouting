import gc
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    cast,
    no_type_check,
    overload,
)

import toml
import torch as t
import transformer_lens as tl
from jaxtyping import Float, Int
from pydantic import BaseModel, field_validator
from tqdm import tqdm
from transformers import BatchEncoding

from ssr import DEVICE, MODELS_PATH, TEMPLATES_PATH
from ssr.defaults import DEFAULT_VALUE, DefaultValue, underload
from ssr.files import load_dataset
from ssr.memory import find_executable_batch_size
from ssr.types import HookList, Tokenizer


class LensDefaults(BaseModel):
    # Name in Transformer Lens
    model_name: Optional[str] = None

    # Surname when calling Lens.from_preset()
    model_surname: Optional[str] = None

    # If padding=False, only keep the sentences that - when tokenized - have a length of seq_len token. Use the Lens.get_max_seq_len util to compute it automatically for a given dataset
    seq_len: Optional[int] = None

    # Max number of samples when fetching/ scanning a dataset
    max_samples: Optional[int] = None

    padding: bool = True
    padding_side: Literal["right", "left"] = "left"

    # We manage special tokens ourselves
    add_special_tokens: bool = False

    # See Transfomer Lens
    pattern: Optional[str] = "resid_post"

    # If None: same as pattern
    stack_act_name: Optional[str] = None

    reduce_seq_method: Literal["last", "mean", "max"] = "last"

    # D_instruction in the paper is "mod"
    dataset_name: Literal["mod", "adv", "mini", "bomb"] = "mod"

    # Either a .jinja2 filepath, or directly as str
    chat_template: Optional[str] = None

    # Don't choose candidates with special tokens like <eos> when optimizing with SSR
    restricted_tokens: Optional[List[str | int]] = None

    # See Transformer Lens's center_unembed, center_writing_weights, and fold_ln
    centered: bool = False

    device: str = DEVICE
    max_tokens_generated: int = 64

    # See Transformer Lens
    fwd_hooks: HookList = []

    truncation: bool = False
    add_generation_prompt: bool = True

    # Used in Lens.apply_chate_template()
    role: str = "user"

    batch_size: int = 62
    system_message: Optional[str] = "You are a helpful assistant."

    @field_validator("system_message", mode="after")
    @classmethod
    def str_none_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value == "none":
            return None
        return value


# See reproduce_experiments/using_lens.ipynb to have a glimpse of
# - How preset can be loaded
# - How the default values are managed
# - How utilities work
class Lens:
    def __init__(
        self,
        model: tl.HookedTransformer,
        default_values: Optional[LensDefaults] = None,
    ):
        self.model = model

        self.defaults = (
            default_values
            if default_values is not None
            else LensDefaults(model_name=model.cfg.model_name)
        )
        self.model_name = cast(str, self.defaults.model_name)

        if self.model.tokenizer is None:
            raise ValueError("model.tokenizer is supposed not None")
        self.tokenizer = cast(Tokenizer, self.model.tokenizer)

    @classmethod
    def from_preset(cls, model_surname: str, **kwargs) -> "Lens":
        """
        Load a preset from the models.toml file. Currently available presets:
            from_preset("llama3.2_1b")
            from_preset("llama3.2_3b")
            from_preset("gemma2_2b")
            from_preset("qwen2.5_1.5b")
        """
        with open(MODELS_PATH, "r") as f:
            data = toml.load(f)

        default_values = LensDefaults(
            **(data.get(model_surname, {}) | kwargs | {"model_surname": model_surname})
        )

        if default_values.centered:
            model = tl.HookedTransformer.from_pretrained(
                model_name=cast(str, default_values.model_name),
                device=default_values.device,
                dtype="float16",
                center_unembed=True,
                center_writing_weights=True,
                fold_ln=True,
            )
        else:
            model = tl.HookedTransformer.from_pretrained_no_processing(
                model_name=cast(str, default_values.model_name),
                device=default_values.device,
                dtype=t.float16,
            )

        if model.tokenizer is None:
            raise ValueError("model.tokenizer is supposed not None")

        if chat_template := default_values.chat_template:
            if ".jinja" in chat_template:
                with open(TEMPLATES_PATH / Path(chat_template), "r") as f:
                    model.tokenizer.chat_template = f.read()

            else:
                model.tokenizer.chat_template = chat_template

        model.tokenizer.padding_side = default_values.padding_side

        return cls(model=model, default_values=default_values)

    # From https://github.com/andyrdt/refusal_direction, @arditi2024refusal, Apache-2.0 license
    @no_type_check
    @underload
    def generate_with_hooks(
        self,
        toks: Int[t.Tensor, "batch_size seq_len"],
        max_tokens_generated: DefaultValue | int = DEFAULT_VALUE,
        fwd_hooks: DefaultValue | HookList = DEFAULT_VALUE,
    ) -> List[str]: ...

    def generate_with_hooks_(
        self,
        toks: Int[t.Tensor, "batch_size seq_len"],
        max_tokens_generated: int,
        fwd_hooks: HookList,
    ) -> List[str]:
        all_toks = (
            t.zeros(toks.shape[0], toks.shape[1] + max_tokens_generated)
            .to(toks.device)
            .long()
        )
        all_toks[:, : toks.shape[1]] = toks

        for i in range(max_tokens_generated):
            with self.model.hooks(fwd_hooks=fwd_hooks):
                logits = self.model(all_toks[:, : -max_tokens_generated + i])
                next_tokens = logits[:, -1, :].argmax(
                    dim=-1
                )  # greedy sampling (temperature=0)

                del logits
                t.cuda.empty_cache()
                gc.collect()

                all_toks[:, -max_tokens_generated + i] = next_tokens

        result = self.model.tokenizer.batch_decode(  # type: ignore
            all_toks[:, toks.shape[1] :], skip_special_tokens=True
        )

        del toks, all_toks
        t.cuda.empty_cache()
        gc.collect()

        return result

    # From https://github.com/andyrdt/refusal_direction, @arditi2024refusal, Apache-2.0 license
    @no_type_check
    @underload
    def get_generations(
        self,
        prompts: List[str] | str,
        padding: DefaultValue | bool = DEFAULT_VALUE,
        truncation: DefaultValue | bool = DEFAULT_VALUE,
        add_special_tokens: DefaultValue | bool = DEFAULT_VALUE,
        max_tokens_generated: DefaultValue | int = DEFAULT_VALUE,
        fwd_hooks: DefaultValue | HookList = DEFAULT_VALUE,
    ) -> List[str]: ...

    @find_executable_batch_size(starting_batch_size=6)
    def get_generations_(
        self,
        prompts: List[str] | str,
        batch_size: int,
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
        max_tokens_generated: int,
        fwd_hooks: HookList,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        generations = []

        for i in tqdm(range(0, len(prompts), batch_size)):
            toks = self.tokenizer(
                prompts[i : i + batch_size],
                padding=padding,
                truncation=truncation,
                add_special_tokens=add_special_tokens,
                return_tensors="pt",
            ).input_ids

            generation = self.generate_with_hooks(
                toks,
                max_tokens_generated=max_tokens_generated,
                fwd_hooks=fwd_hooks,
            )

            del toks
            t.cuda.empty_cache()
            gc.collect()

            generations.extend(generation)

        return generations

    @no_type_check
    @underload
    def auto_scan(
        self,
        inputs: str | List[str] | Int[t.Tensor, "batch_size seq_len"],
        pattern: DefaultValue | Optional[str] = DEFAULT_VALUE,
        padding: DefaultValue | bool = DEFAULT_VALUE,
        truncation: DefaultValue | bool = DEFAULT_VALUE,
        add_special_tokens: DefaultValue | bool = DEFAULT_VALUE,
        layer: Optional[int] = None,
    ) -> Tuple[Float[t.Tensor, "bach seq d_vocab"], tl.ActivationCache]: ...

    @find_executable_batch_size
    def auto_scan_(
        self,
        inputs: str | List[str] | Int[t.Tensor, "batch_size seq_len"],
        batch_size: int,
        pattern: Optional[str],
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
        layer: Optional[int] = None,
    ) -> Tuple[Float[t.Tensor, "bach seq d_vocab"], tl.ActivationCache]:
        """
        Same as Transformer Lens' model.run_with_cache, but with batch to CPU and GPU protection
        """
        if isinstance(inputs, str | list):
            if isinstance(inputs, str):
                inputs = [inputs]
            inputs = self.tokenizer(
                inputs,
                padding=padding,
                truncation=truncation,
                add_special_tokens=add_special_tokens,
                return_tensors="pt",
            ).input_ids

        tokens = cast(t.Tensor, inputs)

        if layer is not None and layer < 0:
            layer += self.model.cfg.n_layers

        base_cache = dict()
        total_samples = tokens.shape[0]

        logits_list = []

        for i in tqdm(range(0, total_samples, batch_size)):
            if pattern is not None:
                logits, cache = self.model.run_with_cache(
                    tokens[i : i + batch_size],
                    names_filter=lambda hook_name: tl.utils.get_act_name(
                        pattern, layer=layer
                    )
                    in hook_name,
                )
            else:
                logits, cache = self.model.run_with_cache(
                    tokens[i : i + batch_size],
                )

            logits_list.append(logits.detach().cpu())  # type: ignore

            cpu_cache = cache.to("cpu")

            if i == 0:
                base_cache = dict(cpu_cache)
            else:
                for key in cpu_cache:
                    base_cache[key] = t.cat([base_cache[key], cpu_cache[key]], dim=0)

            del logits, cache
            t.cuda.empty_cache()
            gc.collect()

        return t.cat(logits_list, dim=0), tl.ActivationCache(base_cache, self.model)

    @overload
    def apply_chat_template(
        self,
        messages: str | List[Dict[str, str]],
        tokenize: Literal[False] = False,
        system_message: DefaultValue | Optional[str] = DEFAULT_VALUE,
        role: DefaultValue | str = DEFAULT_VALUE,
        add_generation_prompt: DefaultValue | bool = DEFAULT_VALUE,
        **kwargs,
    ) -> str: ...

    @overload
    def apply_chat_template(
        self,
        messages: str | List[Dict[str, str]],
        tokenize: Literal[True],
        system_message: DefaultValue | Optional[str] = DEFAULT_VALUE,
        role: DefaultValue | str = DEFAULT_VALUE,
        add_generation_prompt: DefaultValue | bool = DEFAULT_VALUE,
        **kwargs,
    ) -> BatchEncoding: ...

    @no_type_check
    @underload
    def apply_chat_template(
        self,
        messages: str | List[Dict[str, str]],
        tokenize: Literal[True] | Literal[False] = False,
        system_message: DefaultValue | Optional[str] = DEFAULT_VALUE,
        role: DefaultValue | str = DEFAULT_VALUE,
        add_generation_prompt: DefaultValue | bool = DEFAULT_VALUE,
        **kwargs,
    ) -> str | BatchEncoding: ...

    def apply_chat_template_(
        self,
        messages: str | List[Dict[str, str]],
        system_message: Optional[str],
        role: str,
        add_generation_prompt: bool,
        tokenize: Literal[True] | Literal[False] = False,
        **kwargs,
    ) -> str | BatchEncoding:
        # TODO manage return_tensor="pt"
        if isinstance(messages, str):
            messages = [{"role": role, "content": messages}]

        if system_message is not None:
            if "gemma" in self.model.cfg.model_name:
                print(
                    "WARNING: This gemma may not support system message, thus I'll ignore it just in case :)"
                )
            else:
                messages = [{"role": "system", "content": system_message}] + messages

        return cast(
            str | BatchEncoding,
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            ),
        )

    @no_type_check
    @underload
    def process_dataset(
        self,
        hf_raw: List[str],
        hl_raw: List[str],
        padding_side: DefaultValue | Literal["left", "right"] = DEFAULT_VALUE,
        max_samples: DefaultValue | Optional[int] = DEFAULT_VALUE,
        system_message: DefaultValue | Optional[str] = DEFAULT_VALUE,
        add_special_tokens: DefaultValue | bool = DEFAULT_VALUE,
        seq_len: DefaultValue | Optional[int] = DEFAULT_VALUE,
    ) -> Tuple[
        Int[t.Tensor, "batch_size seq_len"], Int[t.Tensor, "batch_size seq_len"]
    ]: ...

    def process_dataset_(
        self,
        hf_raw: List[str],
        hl_raw: List[str],
        padding_side: Literal["left", "right"],
        max_samples: Optional[int],
        system_message: Optional[str],
        add_special_tokens: bool,
        seq_len: Optional[int],
    ) -> Tuple[
        Int[t.Tensor, "batch_size seq_len"], Int[t.Tensor, "batch_size seq_len"]
    ]:
        """
        This routine will:
            - Apply the chat template
            - If padding = True
                - Tokenize with padding
            - If padding = False
                - Tokenize one by one
                - Pick the tokenized sentences that have a length of seq_len
                - Concatenate into one tensor
        """
        max_samples_ = (
            max_samples if max_samples is not None else max(len(hf_raw), len(hl_raw))
        )

        match seq_len:
            case None:
                if padding_side is not None:
                    self.tokenizer.padding_side = padding_side

                if max_samples_ is not None:
                    hf_raw = hf_raw[:max_samples_]
                    hl_raw = hl_raw[:max_samples_]

                hf_ = [
                    self.apply_chat_template(p, system_message=system_message)
                    for p in hf_raw
                ]
                hl_ = [
                    self.apply_chat_template(p, system_message=system_message)
                    for p in hl_raw
                ]

                hf = self.tokenizer(
                    hf_,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=add_special_tokens,
                ).input_ids
                hl = self.tokenizer(
                    hl_,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=add_special_tokens,
                ).input_ids

                return hf, hl

            case _:
                hf_ = [
                    tokens
                    for p in hf_raw
                    if len(
                        tokens := self.apply_chat_template(
                            p, tokenize=True, system_message=system_message
                        )
                    )
                    == seq_len
                ]
                hl_ = [
                    tokens
                    for p in hl_raw
                    if len(
                        tokens := self.apply_chat_template(
                            p, tokenize=True, system_message=system_message
                        )
                    )
                    == seq_len
                ]

                min_len = min(len(hf_), len(hl_), max_samples_)
                hf_ = hf_[:min_len]
                hl_ = hl_[:min_len]

                hf = t.cat([t.Tensor(p).unsqueeze(0).long() for p in hf_], dim=0)
                hl = t.cat([t.Tensor(p).unsqueeze(0).long() for p in hl_], dim=0)

                return hf, hl

    @no_type_check
    @underload
    def scan_dataset(
        self,
        hf: Int[t.Tensor, "batch_size seq_len"],
        hl: Int[t.Tensor, "batch_size seq_len"],
        pattern: DefaultValue | str = DEFAULT_VALUE,
        reduce_seq_method: DefaultValue
        | Literal["last", "mean", "max"] = DEFAULT_VALUE,
        stack_act_name: DefaultValue | Optional[str] = DEFAULT_VALUE,
    ) -> Tuple[
        Float[t.Tensor, "n_layers batch_size d_model"],
        Float[t.Tensor, "n_layers batch_size d_model"],
    ]: ...

    def scan_dataset_(
        self,
        hf: Int[t.Tensor, "batch_size seq_len"],
        hl: Int[t.Tensor, "batch_size seq_len"],
        pattern: str,
        reduce_seq_method: Literal["last", "mean", "max"],
        stack_act_name: Optional[str],
    ) -> Tuple[
        Float[t.Tensor, "n_layers batch_size d_model"],
        Float[t.Tensor, "n_layers batch_size d_model"],
    ]:
        """
        This routine will take tokenized inputs, scan and apply the reduce method
        """
        _, hf_scan = self.auto_scan(hf, pattern=pattern)
        _, hl_scan = self.auto_scan(hl, pattern=pattern)
        stack_act_name_ = stack_act_name if stack_act_name is not None else pattern

        try:
            hf_act = hf_scan.stack_activation(stack_act_name_)
            hl_act = hl_scan.stack_activation(stack_act_name_)
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

    @no_type_check
    @underload
    def auto_scan_dataset(
        self,
        dataset_name: DefaultValue
        | Literal["mod", "adv", "mini", "bomb"] = DEFAULT_VALUE,
        max_samples: DefaultValue | Optional[int] = DEFAULT_VALUE,
        padding: DefaultValue | bool = DEFAULT_VALUE,
        padding_side: DefaultValue | Literal["left", "right"] = DEFAULT_VALUE,
        system_message: DefaultValue | Optional[str] = DEFAULT_VALUE,
        reduce_seq_method: DefaultValue
        | Literal["last", "mean", "max"] = DEFAULT_VALUE,
        pattern: DefaultValue | str = DEFAULT_VALUE,
        stack_act_name: DefaultValue | Optional[str] = DEFAULT_VALUE,
    ) -> Tuple[
        Float[t.Tensor, "n_layers batch_size d_model"],
        Float[t.Tensor, "n_layers batch_size d_model"],
    ]: ...

    def auto_scan_dataset_(
        self,
        dataset_name: Literal["mod", "adv", "mini", "bomb"],
        max_samples: Optional[int],
        padding: bool,
        padding_side: Literal["left", "right"],
        system_message: Optional[str],
        reduce_seq_method: Literal["last", "mean", "max"],
        pattern: str,
        stack_act_name: Optional[str],
    ) -> Tuple[
        Float[t.Tensor, "n_layers batch_size d_model"],
        Float[t.Tensor, "n_layers batch_size d_model"],
    ]:
        """
        This routine will:
            - Load the dataset
            - Compute the seq_len needed if padding = False
            - Call process_dataset and scan_dataset
        """
        if max_samples is not None:
            hf_raw, hl_raw = load_dataset(dataset_name, max_samples=max_samples)
        else:
            hf_raw, hl_raw = load_dataset(dataset_name)

        seq_len_ = self.get_max_seq_len(hf_raw, hl_raw)[0] if not padding else None

        hf, hl = self.process_dataset(
            hf_raw,
            hl_raw,
            max_samples=max_samples,
            padding_side=padding_side,
            system_message=system_message,
            seq_len=seq_len_,
        )

        return self.scan_dataset(
            hf,
            hl,
            pattern=pattern,
            stack_act_name=stack_act_name,
            reduce_seq_method=reduce_seq_method,
        )

    @no_type_check
    @underload
    def get_max_seq_len(
        self,
        a_raw: List[str],
        b_raw: List[str],
        system_message: DefaultValue | Optional[str] = DEFAULT_VALUE,
        role: DefaultValue | str = DEFAULT_VALUE,
        add_generation_prompt: DefaultValue | bool = DEFAULT_VALUE,
        **kwargs,
    ) -> Tuple[int, int]: ...

    def get_max_seq_len_(
        self,
        a_raw: List[str],
        b_raw: List[str],
        system_message: Optional[str],
        role: str,
        add_generation_prompt: bool,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Returns:
            Tuple[int, int]: seq_len, n_samples
        """
        from collections import Counter

        a_tokens = [
            self.apply_chat_template(
                x,
                tokenize=True,
                system_message=system_message,
                role=role,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            for x in a_raw
        ]
        b_tokens = [
            self.apply_chat_template(
                x,
                tokenize=True,
                system_message=system_message,
                role=role,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            for x in b_raw
        ]

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
