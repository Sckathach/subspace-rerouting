from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, cast, overload

import torch as t
import transformer_lens as tl
from accelerate.utils.memory import find_executable_batch_size, release_memory
from jaxtyping import Float, Int
from pydantic import BaseModel
from tqdm import tqdm
from transformers import BatchEncoding

from ssr import DEVICE, TEMPLATES_PATH
from ssr.files import load_dataset
from ssr.types import HookList, Tokenizer


class DefaultValue:
    pass


DEFAULT_VALUE = DefaultValue()


class Values(BaseModel):
    model_name: Optional[str] = None
    surname: Optional[str] = None
    seq_len: Optional[int] = None
    max_samples: Optional[int] = None
    padding: bool = True
    padding_side: Literal["right", "left"] = "left"
    add_special_tokens: bool = False
    pattern: str = "resid_post"
    scan_pattern: Optional[str] = None
    stack_act_name: Optional[str] = None  # if None: same as pattern
    reduce_seq_method: Literal["last", "mean", "max"] = "last"
    dataset_name: Literal["mod", "adv", "mini", "bomb"] = "mod"
    chat_template: Optional[str] = None
    restricted_tokens: Optional[List[str | int]] = None
    centered: bool = False
    device: str = DEVICE
    max_tokens_generated: int = 64
    fwd_hooks: HookList = []
    generation_batch_size: int = 4
    truncation: bool = False
    add_generation_prompt: bool = True
    role: str = "user"
    batch_size: int = 62
    system_message: Optional[str] = "You are a helpful assistant."


class Llama3_2__1b_Values(Values):
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    surname = "llama3.2_1b"
    chat_template = "llama3.2.jinja2"
    restricted_tokens = ["128000-128255", "non-ascii"]
    # idx between 128000 and 128255 (https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/tokenizer_config.json)


class Llama3_2__3b_Values(Values):
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    surname = "llama3.2_3b"
    chat_template = "llama3.2.jinja2"
    restricted_tokens = ["128000-128255", "non-ascii"]
    # idx between 128000 and 128255 (https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/tokenizer_config.json)


class Gemma2__2b_Values(Values):
    model_name = "google/gemma-2-2b-it"
    surname = "gemma2_2b"
    padding = False
    system_message = None
    chat_template = "gemma2_2b.jinja2"
    restricted_tokens = ["0-108", "non-ascii"]
    # 108 first idx (https://huggingface.co/google/gemma-2-2b-it/blob/main/tokenizer_config.json)


class Qwen2_5__1_5b_Values(Values):
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    surname = "qwen2.5_1.5b"
    chat_template = "qwen2.5_1.5b.jinja2"
    restricted_tokens = ["non-ascii"]


class Lens:
    def __init__(
        self,
        model: tl.HookedTransformer,
        default_values: Optional[Values] = None,
    ):
        self.model = model

        self.default_values = (
            default_values
            if default_values is not None
            else Values(model_name=model.cfg.model_name)
        )
        self.model_name = cast(str, self.default_values.model_name)

        if self.model.tokenizer is None:
            raise ValueError("model.tokenizer is supposed not None")
        self.tokenizer = cast(Tokenizer, self.model.tokenizer)

    def get_default_values(self, **kwargs) -> Values:
        values = self.default_values.model_dump()
        for key, value in kwargs.items():
            if not isinstance(value, DefaultValue):
                values[key] = value
        return Values(**values)

    @classmethod
    def from_config(cls, model_name: str, device: str = DEVICE, **kwargs) -> "Lens":
        model = get_hooked_transformer_from_config(
            model_name=model_name,
            centered=kwargs.get("centered", False),
            device=device,
            **kwargs,
        )

        return cls(model=model)

    # From https://github.com/andyrdt/refusal_direction, @arditi2024refusal, Apache-2.0 license
    def _generate_with_hooks(
        self,
        toks: Int[t.Tensor, "batch_size seq_len"],
        max_tokens_generated: DefaultValue | int = DEFAULT_VALUE,
        fwd_hooks: DefaultValue | HookList = DEFAULT_VALUE,
    ) -> List[str]:
        args = self.get_default_values(
            max_tokens_generated=max_tokens_generated, fwd_hooks=fwd_hooks
        )

        all_toks = (
            t.zeros(toks.shape[0], toks.shape[1] + args.max_tokens_generated)
            .to(toks.device)
            .long()
        )
        all_toks[:, : toks.shape[1]] = toks

        for i in range(args.max_tokens_generated):
            with self.model.hooks(fwd_hooks=args.fwd_hooks):
                logits = self.model(all_toks[:, : -args.max_tokens_generated + i])
                next_tokens = logits[:, -1, :].argmax(
                    dim=-1
                )  # greedy sampling (temperature=0)

                logits = release_memory(logits)
                all_toks[:, -args.max_tokens_generated + i] = next_tokens

        result = self.model.tokenizer.batch_decode(  # type: ignore
            all_toks[:, toks.shape[1] :], skip_special_tokens=True
        )

        toks, all_toks = release_memory(toks, all_toks)
        return result

    # From https://github.com/andyrdt/refusal_direction, @arditi2024refusal, Apache-2.0 license
    def get_generations(
        self,
        prompts: List[str] | str,
        batch_size: DefaultValue | int = DEFAULT_VALUE,
        padding: DefaultValue | bool = DEFAULT_VALUE,
        truncation: DefaultValue | bool = DEFAULT_VALUE,
        add_special_tokens: DefaultValue | bool = DEFAULT_VALUE,
        max_tokens_generated: DefaultValue | int = DEFAULT_VALUE,
        fwd_hooks: DefaultValue | HookList = DEFAULT_VALUE,
    ) -> List[str]:
        args = self.get_default_values(
            generation_batch_size=batch_size,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )

        if isinstance(prompts, str):
            prompts = [prompts]

        generations = []

        for i in tqdm(range(0, len(prompts), args.generation_batch_size)):
            toks = self.tokenizer(
                prompts[i : i + args.generation_batch_size],
                padding=args.padding,
                truncation=args.truncation,
                add_special_tokens=args.add_special_tokens,
                return_tensors="pt",
            ).input_ids

            generation = self._generate_with_hooks(
                toks,
                max_tokens_generated=args.max_tokens_generated,
                fwd_hooks=args.fwd_hooks,
            )
            toks = release_memory(toks)
            generations.extend(generation)

        return generations

    @overload
    def batch_scan_to_cpu(
        self,
        tokens: Int[t.Tensor, "batch_size seq_len"],
        batch_size: int,
        return_logits: Literal[False] = False,
        scan_pattern: DefaultValue | Optional[str] = DEFAULT_VALUE,
        layer: Optional[int] = None,
    ) -> tl.ActivationCache: ...

    @overload
    def batch_scan_to_cpu(
        self,
        tokens: Int[t.Tensor, "batch_size seq_len"],
        batch_size: int,
        return_logits: Literal[True],
        scan_pattern: DefaultValue | Optional[str] = DEFAULT_VALUE,
        layer: Optional[int] = None,
    ) -> Tuple[Float[t.Tensor, "batch seq d_vocab"], tl.ActivationCache]: ...

    def batch_scan_to_cpu(
        self,
        tokens: Int[t.Tensor, "batch_size seq_len"],
        batch_size: int,
        return_logits: Literal[False] | Literal[True] = False,
        scan_pattern: DefaultValue | Optional[str] = DEFAULT_VALUE,
        layer: Optional[int] = None,
    ) -> (
        tl.ActivationCache
        | Tuple[Float[t.Tensor, "bach seq d_vocab"], tl.ActivationCache]
    ):
        if layer is not None and layer < 0:
            layer += self.model.cfg.n_layers
        args = self.get_default_values(scan_pattern=scan_pattern)

        base_cache = dict()
        total_samples = tokens.shape[0]

        logits_list = []

        for i in tqdm(range(0, total_samples, batch_size)):
            if args.scan_pattern is not None:
                logits, cache = self.model.run_with_cache(
                    tokens[i : i + batch_size],
                    names_filter=lambda hook_name: tl.utils.get_act_name(
                        cast(str, args.scan_pattern), layer=layer
                    )
                    in hook_name,
                )
            else:
                logits, cache = self.model.run_with_cache(
                    tokens[i : i + batch_size],
                )

            if return_logits:
                logits_list.append(logits.detach().cpu())  # type: ignore

            cpu_cache = cache.to("cpu")

            if i == 0:
                base_cache = dict(cpu_cache)
            else:
                for key in cpu_cache:
                    base_cache[key] = t.cat([base_cache[key], cpu_cache[key]], dim=0)

            logits, cache = release_memory(logits, cache)

        if return_logits:
            return t.cat(logits_list, dim=0), tl.ActivationCache(base_cache, self.model)

        return tl.ActivationCache(base_cache, self.model)

    @overload
    def auto_scan(
        self,
        inputs: str | List[str] | Int[t.Tensor, "batch_size seq_len"],
        return_logits: Literal[False] = False,
        layer: Optional[int] = None,
        starting_batch_size: Optional[int] = None,
        scan_pattern: DefaultValue | Optional[str] = DEFAULT_VALUE,
        padding: DefaultValue | bool = DEFAULT_VALUE,
        truncation: DefaultValue | bool = DEFAULT_VALUE,
        add_special_tokens: DefaultValue | bool = DEFAULT_VALUE,
    ) -> tl.ActivationCache: ...

    @overload
    def auto_scan(
        self,
        inputs: str | List[str] | Int[t.Tensor, "batch_size seq_len"],
        return_logits: Literal[True],
        layer: Optional[int] = None,
        starting_batch_size: Optional[int] = None,
        scan_pattern: DefaultValue | Optional[str] = DEFAULT_VALUE,
        padding: DefaultValue | bool = DEFAULT_VALUE,
        truncation: DefaultValue | bool = DEFAULT_VALUE,
        add_special_tokens: DefaultValue | bool = DEFAULT_VALUE,
    ) -> Tuple[Float[t.Tensor, "batch seq d_vocab"], tl.ActivationCache]: ...

    def auto_scan(
        self,
        inputs: str | List[str] | Int[t.Tensor, "batch_size seq_len"],
        return_logits: Literal[False] | Literal[True] = False,
        layer: Optional[int] = None,
        starting_batch_size: Optional[int] = None,
        scan_pattern: DefaultValue | Optional[str] = DEFAULT_VALUE,
        padding: DefaultValue | bool = DEFAULT_VALUE,
        truncation: DefaultValue | bool = DEFAULT_VALUE,
        add_special_tokens: DefaultValue | bool = DEFAULT_VALUE,
    ) -> (
        tl.ActivationCache
        | Tuple[Float[t.Tensor, "batch seq d_vocab"], tl.ActivationCache]
    ):
        args = self.get_default_values(
            scan_pattern=scan_pattern,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            padding=padding,
        )
        if isinstance(inputs, str | list):
            if isinstance(inputs, str):
                inputs = [inputs]
            inputs = self.tokenizer(
                inputs,
                padding=args.padding,
                truncation=args.truncation,
                add_special_tokens=args.add_special_tokens,
                return_tensors="pt",
            ).input_ids

        def auto_batch_scan_to_cpu(search_batch_size: int, reduced_inputs: t.Tensor):
            return self.batch_scan_to_cpu(
                reduced_inputs,
                return_logits=return_logits,  # type: ignore
                scan_pattern=args.scan_pattern,
                layer=layer,
                batch_size=search_batch_size,
            )

        if starting_batch_size is None:
            starting_batch_size = len(inputs)

        if return_logits:
            return cast(
                Tuple[Float[t.Tensor, "batch seq d_vocab"], tl.ActivationCache],
                find_executable_batch_size(auto_batch_scan_to_cpu, starting_batch_size)(
                    inputs
                ),
            )
        return cast(
            tl.ActivationCache,
            find_executable_batch_size(auto_batch_scan_to_cpu, starting_batch_size)(
                inputs
            ),
        )

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

    def apply_chat_template(
        self,
        messages: str | List[Dict[str, str]],
        tokenize: Literal[True] | Literal[False] = False,
        system_message: DefaultValue | Optional[str] = DEFAULT_VALUE,
        role: DefaultValue | str = DEFAULT_VALUE,
        add_generation_prompt: DefaultValue | bool = DEFAULT_VALUE,
        **kwargs,
    ) -> str | BatchEncoding:
        # TODO manage return_tensor="pt"
        args = self.get_default_values(
            system_message=system_message,
            role=role,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(messages, str):
            messages = [{"role": args.role, "content": messages}]

        if args.system_message is not None:
            if "gemma" in self.model.cfg.model_name:
                print(
                    "WARNING: This gemma may not support system message, thus I'll ignore it just in case :)"
                )
            else:
                messages = [
                    {"role": "system", "content": args.system_message}
                ] + messages

        return cast(
            str | BatchEncoding,
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=args.add_generation_prompt,
                **kwargs,
            ),
        )

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
    ]:
        args = self.get_default_values(
            padding_side=padding_side,
            max_samples=max_samples,
            system_message=system_message,
            add_special_tokens=add_special_tokens,
            seq_len=seq_len,
        )
        max_samples_ = (
            args.max_samples
            if args.max_samples is not None
            else max(len(hf_raw), len(hl_raw))
        )

        match args.seq_len:
            case None:
                if args.padding_side is not None:
                    self.tokenizer.padding_side = args.padding_side

                if max_samples_ is not None:
                    hf_raw = hf_raw[:max_samples_]
                    hl_raw = hl_raw[:max_samples_]

                hf_ = [
                    self.apply_chat_template(p, system_message=args.system_message)
                    for p in hf_raw
                ]
                hl_ = [
                    self.apply_chat_template(p, system_message=args.system_message)
                    for p in hl_raw
                ]

                hf = self.tokenizer(
                    hf_,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=args.add_special_tokens,
                ).input_ids
                hl = self.tokenizer(
                    hl_,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=args.add_special_tokens,
                ).input_ids

                return hf, hl

            case _:
                hf_ = [
                    tokens
                    for p in hf_raw
                    if len(
                        tokens := self.apply_chat_template(
                            p, tokenize=True, system_message=args.system_message
                        )
                    )
                    == args.seq_len
                ]
                hl_ = [
                    tokens
                    for p in hl_raw
                    if len(
                        tokens := self.apply_chat_template(
                            p, tokenize=True, system_message=args.system_message
                        )
                    )
                    == args.seq_len
                ]

                min_len = min(len(hf_), len(hl_), max_samples_)
                hf_ = hf_[:min_len]
                hl_ = hl_[:min_len]

                hf = t.cat([t.Tensor(p).unsqueeze(0).long() for p in hf_], dim=0)
                hl = t.cat([t.Tensor(p).unsqueeze(0).long() for p in hl_], dim=0)

                return hf, hl

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
    ]:
        args = self.get_default_values(
            pattern=pattern,
            reduce_seq_method=reduce_seq_method,
            stack_act_name=stack_act_name,
        )
        hf_scan = self.auto_scan(hf, scan_pattern=args.pattern)
        hl_scan = self.auto_scan(hl, scan_pattern=args.pattern)
        stack_act_name_ = (
            args.stack_act_name if args.stack_act_name is not None else args.pattern
        )

        try:
            hf_act = hf_scan.stack_activation(stack_act_name_)
            hl_act = hl_scan.stack_activation(stack_act_name_)
        except Exception as e:
            raise ValueError(
                f"Cannot stack activations! Check stack_act_name. Error: {e}"
            ) from e

        match args.reduce_seq_method:
            case "last":
                return hf_act[:, :, -1, :], hl_act[:, :, -1, :]
            case "mean":
                return hf_act.mean(dim=2), hl_act.mean(dim=2)
            case "max":
                return hf_act.max(dim=2)[0], hl_act.max(dim=2)[0]

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
        seq_len: DefaultValue | Optional[int] = DEFAULT_VALUE,
        pattern: DefaultValue | str = DEFAULT_VALUE,
        stack_act_name: DefaultValue | Optional[str] = DEFAULT_VALUE,
    ) -> Tuple[
        Float[t.Tensor, "n_layers batch_size d_model"],
        Float[t.Tensor, "n_layers batch_size d_model"],
    ]:
        args = self.get_default_values(
            dataset_name=dataset_name,
            max_samples=max_samples,
            padding=padding,
            padding_side=padding_side,
            system_message=system_message,
            reduce_seq_method=reduce_seq_method,
            seq_len=seq_len,
            pattern=pattern,
            stack_act_name=stack_act_name,
        )
        hf_raw, hl_raw = load_dataset(args.dataset_name)
        seq_len_ = self.get_max_seq_len(hf_raw, hl_raw)[0] if not args.padding else None

        hf, hl = self.process_dataset(
            hf_raw,
            hl_raw,
            max_samples=args.max_samples,
            padding_side=args.padding_side,
            system_message=args.system_message,
            seq_len=seq_len_,
        )

        return self.scan_dataset(
            hf,
            hl,
            pattern=args.pattern,
            stack_act_name=args.stack_act_name,
            reduce_seq_method=args.reduce_seq_method,
        )

    def get_max_seq_len(
        self,
        a_raw: List[str],
        b_raw: List[str],
        system_message: DefaultValue | Optional[str] = DEFAULT_VALUE,
        role: DefaultValue | str = DEFAULT_VALUE,
        add_generation_prompt: DefaultValue | bool = DEFAULT_VALUE,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Returns:
            Tuple[int, int]: seq_len, n_samples
        """
        from collections import Counter

        args = self.get_default_values(
            system_message=system_message,
            role=role,
            add_generation_prompt=add_generation_prompt,
        )

        a_tokens = [
            self.apply_chat_template(
                x,
                tokenize=True,
                system_message=args.system_message,
                role=args.role,
                add_generation_prompt=args.add_generation_prompt,
                **kwargs,
            )
            for x in a_raw
        ]
        b_tokens = [
            self.apply_chat_template(
                x,
                tokenize=True,
                system_message=args.system_message,
                role=args.role,
                add_generation_prompt=args.add_generation_prompt,
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


def get_hooked_transformer_from_config(
    model_name: str, default_values: Optional[Values] = None, **kwargs
) -> tl.HookedTransformer:
    models = {
        "llama3.2_1b": Llama3_2__1b_Values(),
        "llama3.2_3b": Llama3_2__3b_Values(),
        "gemma2_2b": Gemma2__2b_Values(),
        "qwen2.5_1.5b": Qwen2_5__1_5b_Values(),
    }
    if default_values_ := models.get(model_name, default_values):
        if default_values_.centered:
            model = tl.HookedTransformer.from_pretrained(
                model_name=default_values_.model_name,
                device=default_values_.device,
                dtype="float16",
                center_unembed=kwargs.get("center_unembed", True),
                center_writing_weights=kwargs.get("center_writing_weights", True),
                fold_ln=kwargs.get("fold_ln", True),
                **kwargs,
            )
        else:
            model = tl.HookedTransformer.from_pretrained_no_processing(
                model_name=default_values_.model_name,
                device=default_values_.device,
                dtype=t.float16,
            )

        if model.tokenizer is None:
            raise ValueError("model.tokenizer is supposed not None")

        if chat_template := kwargs.get("chat_template"):
            if ".jinja" in chat_template:
                with open(TEMPLATES_PATH / Path(chat_template), "r") as f:
                    model.tokenizer.chat_template = f.read()

            else:
                model.tokenizer.chat_template = chat_template

        if padding_side := kwargs.get("padding_side"):
            model.tokenizer.padding_side = padding_side

        if pad_token := kwargs.get("pad_token"):
            model.tokenizer.pad_token = pad_token

        return model
    raise ValueError(
        f"default_values can not be None if model_name is not in: {models.keys()}"
    )
