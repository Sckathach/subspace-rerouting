from typing import Callable, List, Tuple, Union

import torch as t
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

HookFunction = Callable[
    [Float[t.Tensor, "... d_act"], HookPoint], Float[t.Tensor, "... d_act"]
]
HookName = Union[str, Callable[[str], bool]]
HookList = List[Tuple[HookName, HookFunction]]

Tokenizer = PreTrainedTokenizerFast | PreTrainedTokenizer

Optimizer = t.optim.Optimizer
Loss = t.nn.modules.loss._Loss
