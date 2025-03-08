import gc
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, cast, overload

import toml
import torch as t
import transformer_lens as tl
from jaxtyping import Float, Int
from tqdm import tqdm
from transformers import BatchEncoding

from ssr import DEVICE, MODELS_PATH, TEMPLATES_PATH
from ssr.memory import find_executable_batch_size
from ssr.types import HookList, Tokenizer


class Lens:
    def __init__(self, model: tl.HookedTransformer):
        self.model = model
        self.model_name = get_surname(self.model.cfg.model_name)

        if self.model.tokenizer is None:
            raise ValueError("model.tokenizer is supposed not None")
        self.tokenizer = cast(Tokenizer, self.model.tokenizer)

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
        max_tokens_generated: int = 64,
        fwd_hooks: Optional[HookList] = None,
    ) -> List[str]:
        if fwd_hooks is None:
            fwd_hooks = []

        all_toks = t.zeros(
            (toks.shape[0], toks.shape[1] + max_tokens_generated),
            dtype=t.long,
            device=toks.device,
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
    def get_generations(
        self,
        prompts: List[str] | str,
        fwd_hooks: Optional[HookList] = None,
        max_tokens_generated: int = 64,
        batch_size: int = 4,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = False,
    ) -> List[str]:
        if fwd_hooks is None:
            fwd_hooks = []

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

            generation = self._generate_with_hooks(
                toks,
                max_tokens_generated=max_tokens_generated,
                fwd_hooks=fwd_hooks,
            )

            del toks
            t.cuda.empty_cache()
            gc.collect()
            generations.extend(generation)

        return generations

    @overload
    def batch_scan_to_cpu(
        self,
        tokens: Int[t.Tensor, "batch_size seq_len"],
        batch_size: int,
        return_logits: Literal[False] = False,
        pattern: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> tl.ActivationCache: ...

    @overload
    def batch_scan_to_cpu(
        self,
        tokens: Int[t.Tensor, "batch_size seq_len"],
        batch_size: int,
        return_logits: Literal[True],
        pattern: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> Tuple[Float[t.Tensor, "batch seq d_vocab"], tl.ActivationCache]: ...

    def batch_scan_to_cpu(
        self,
        tokens: Int[t.Tensor, "batch_size seq_len"],
        batch_size: int,
        return_logits: Literal[False] | Literal[True] = False,
        pattern: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> (
        tl.ActivationCache
        | Tuple[Float[t.Tensor, "bach seq d_vocab"], tl.ActivationCache]
    ):
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

            if return_logits:
                logits_list.append(logits.detach().cpu())  # type: ignore

            cpu_cache = cache.to("cpu")

            if i == 0:
                base_cache = dict(cpu_cache)
            else:
                for key in cpu_cache:
                    base_cache[key] = t.cat([base_cache[key], cpu_cache[key]], dim=0)

            del logits, cache
            gc.collect()
            t.cuda.empty_cache()

        if return_logits:
            return t.cat(logits_list, dim=0), tl.ActivationCache(base_cache, self.model)

        return tl.ActivationCache(base_cache, self.model)

    @overload
    def auto_scan(
        self,
        inputs: str | List[str] | Int[t.Tensor, "batch_size seq_len"],
        return_logits: Literal[False] = False,
        pattern: Optional[str] = None,
        layer: Optional[int] = None,
        starting_batch_size: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = False,
    ) -> tl.ActivationCache: ...

    @overload
    def auto_scan(
        self,
        inputs: str | List[str] | Int[t.Tensor, "batch_size seq_len"],
        return_logits: Literal[True],
        pattern: Optional[str] = None,
        layer: Optional[int] = None,
        starting_batch_size: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = False,
    ) -> Tuple[Float[t.Tensor, "batch seq d_vocab"], tl.ActivationCache]: ...

    def auto_scan(
        self,
        inputs: str | List[str] | Int[t.Tensor, "batch_size seq_len"],
        return_logits: Literal[False] | Literal[True] = False,
        pattern: Optional[str] = None,
        layer: Optional[int] = None,
        starting_batch_size: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = False,
    ) -> (
        tl.ActivationCache
        | Tuple[Float[t.Tensor, "batch seq d_vocab"], tl.ActivationCache]
    ):
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

        def auto_batch_scan_to_cpu(search_batch_size: int, reduced_inputs: t.Tensor):
            return self.batch_scan_to_cpu(
                reduced_inputs,
                return_logits=return_logits,  # type: ignore
                pattern=pattern,
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
        add_generation_prompt: bool = True,
        system_message: Optional[str] = None,
        role: str = "user",
        **kwargs,
    ) -> str: ...

    @overload
    def apply_chat_template(
        self,
        messages: str | List[Dict[str, str]],
        tokenize: Literal[True],
        add_generation_prompt: bool = True,
        system_message: Optional[str] = None,
        role: str = "user",
        **kwargs,
    ) -> BatchEncoding: ...

    def apply_chat_template(
        self,
        messages: str | List[Dict[str, str]],
        tokenize: Literal[True] | Literal[False] = False,
        add_generation_prompt: bool = True,
        system_message: Optional[str] = None,
        role: str = "user",
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


def get_hooked_transformer_from_config(
    model_name: str, centered: bool = False, device: str = DEVICE, **kwargs
) -> tl.HookedTransformer:
    kwargs = kwargs | model_info(model_name)

    if centered:
        model = tl.HookedTransformer.from_pretrained(
            model_name=kwargs["model_name"],
            device=device,
            dtype="float16",
            center_unembed=kwargs.get("center_unembed", True),
            center_writing_weights=kwargs.get("center_writing_weights", True),
            fold_ln=kwargs.get("fold_ln", True),
        )
    else:
        model = tl.HookedTransformer.from_pretrained_no_processing(
            model_name=kwargs["model_name"], device=device, dtype=t.float16
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


def model_info(surname: str) -> Dict[str, str | List[str | int]]:
    with open(MODELS_PATH, "r") as f:
        models = toml.load(f)
    return models.get(surname, {})


def get_surname(model_name: str) -> str:
    with open(MODELS_PATH, "r") as f:
        models = toml.load(f)

    for k, v in models.items():
        if v.get("model_name", "") == model_name or model_name in v.get(
            "other_names", []
        ):
            return k

    raise ValueError(
        f"Surname not found for {model_name} in {MODELS_PATH}. Use the other_names field if necessary."
    )
