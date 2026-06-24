from __future__ import annotations

import torch
import torch.nn as nn


EMBEDDING_INIT_STD = 0.02


def init_embedding(
    module: nn.Embedding,
    *,
    std: float = EMBEDDING_INIT_STD,
    zero_padding_idx: bool = False,
) -> None:
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if zero_padding_idx and module.padding_idx is not None:
        with torch.no_grad():
            module.weight[module.padding_idx].zero_()


def init_linear_xavier(module: nn.Linear) -> None:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_module_xavier(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        init_linear_xavier(module)


def init_parameter_normal(param: nn.Parameter, *, std: float = EMBEDDING_INIT_STD) -> None:
    nn.init.normal_(param, mean=0.0, std=std)


def zero_parameter(param: nn.Parameter) -> None:
    with torch.no_grad():
        param.zero_()
