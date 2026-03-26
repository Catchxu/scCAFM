from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .sfm import FactorState


@dataclass
class FoundationModuleOutput:
    """
    Standardized output for foundation modules.

    Attributes:
    - `raw`: raw module output before wrapper normalization
    - `grn`: optional per-cell GRN tensor
    - `factors`: optional latent factor assignments
    """

    raw: Any
    grn: Optional[torch.Tensor]
    factors: Optional[FactorState]


@dataclass
class ModelWrapperOutput:
    """
    Aggregated outputs from named foundation modules and named heads.

    Attributes:
    - `foundations`: maps foundation module name to its standardized output
    - `heads`: maps head name to the corresponding head output
    """

    foundations: dict[str, FoundationModuleOutput]
    heads: dict[str, Any]


class ModelWrapper(nn.Module):
    """
    Generic composition wrapper for foundation modules and downstream heads.

    This wrapper is designed to scale beyond a single `SFM` and a single head:
    - multiple foundation modules can be registered by name
    - multiple heads can be registered by name
    - each head can be attached to a specific foundation module via
      `head_to_foundation`

    Current expected foundation-module behavior:
    - forward(tokens, return_factors=..., compute_grn=...) -> `grn` or `(grn, factors)`

    Current expected head behavior:
    - heads may accept any subset of:
      `tokens`, `factors`, `grn`, `foundation_output`, `foundation_name`
    """

    def __init__(
        self,
        foundation_modules: Mapping[str, nn.Module],
        head_modules: Optional[Mapping[str, nn.Module]] = None,
        head_to_foundation: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__()

        if len(foundation_modules) == 0:
            raise ValueError("`foundation_modules` must contain at least one module.")

        self.foundation_modules = nn.ModuleDict(foundation_modules)
        self.head_modules = nn.ModuleDict(head_modules or {})
        self.head_to_foundation = dict(head_to_foundation or {})

        for head_name, foundation_name in self.head_to_foundation.items():
            if head_name not in self.head_modules:
                raise KeyError(
                    f"`head_to_foundation` references unknown head {head_name!r}."
                )
            if foundation_name not in self.foundation_modules:
                raise KeyError(
                    f"`head_to_foundation[{head_name!r}]` references unknown "
                    f"foundation module {foundation_name!r}."
                )

        default_foundation = self.default_foundation_name
        for head_name in self.head_modules:
            self.head_to_foundation.setdefault(head_name, default_foundation)

    @property
    def default_foundation_name(self) -> str:
        return next(iter(self.foundation_modules.keys()))

    @staticmethod
    def _normalize_bool_selector(
        value: bool | Mapping[str, bool],
        names: list[str],
        option_name: str,
    ) -> dict[str, bool]:
        if isinstance(value, bool):
            return {name: value for name in names}

        resolved: dict[str, bool] = {name: False for name in names}
        for name, flag in value.items():
            if name not in resolved:
                raise KeyError(
                    f"`{option_name}` references unknown module name {name!r}."
                )
            resolved[name] = bool(flag)
        return resolved

    @staticmethod
    def _normalize_name_subset(
        selected: Optional[list[str]],
        available: nn.ModuleDict,
        subset_name: str,
    ) -> list[str]:
        if selected is None:
            return list(available.keys())

        unknown = [name for name in selected if name not in available]
        if unknown:
            raise KeyError(
                f"Unknown {subset_name} name(s): {unknown}. "
                f"Available names: {list(available.keys())}."
            )
        return selected

    @staticmethod
    def _coerce_foundation_output(
        output: Any,
        *,
        return_factors: bool,
        compute_grn: bool,
    ) -> FoundationModuleOutput:
        if return_factors:
            if not (isinstance(output, tuple) and len(output) == 2):
                raise TypeError(
                    "Expected foundation module to return `(grn, factors)` when "
                    "`return_factors=True`."
                )
            grn, factors = output
            if factors is not None and not isinstance(factors, FactorState):
                raise TypeError(
                    f"Expected `factors` to be `FactorState` or None, got {type(factors).__name__}."
                )
            return FoundationModuleOutput(raw=output, grn=grn, factors=factors)

        expected_grn = output if compute_grn else None
        return FoundationModuleOutput(raw=output, grn=expected_grn, factors=None)

    @staticmethod
    def _build_head_kwargs(
        head: nn.Module,
        *,
        tokens: dict[str, torch.Tensor | None],
        foundation_output: FoundationModuleOutput,
        foundation_name: str,
    ) -> dict[str, Any]:
        candidate_kwargs = {
            "tokens": tokens,
            "factors": foundation_output.factors,
            "grn": foundation_output.grn,
            "foundation_output": foundation_output,
            "foundation_name": foundation_name,
            "module_output": foundation_output,
            "module_name": foundation_name,
        }

        signature = inspect.signature(head.forward)
        parameters = signature.parameters

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return candidate_kwargs

        accepted_names = {
            name
            for name, param in parameters.items()
            if param.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        return {
            key: value
            for key, value in candidate_kwargs.items()
            if key in accepted_names
        }

    def forward(
        self,
        tokens: dict[str, torch.Tensor | None],
        foundation_names: Optional[list[str]] = None,
        head_names: Optional[list[str]] = None,
        compute_grn: bool | Mapping[str, bool] = False,
        return_factors: bool | Mapping[str, bool] = True,
    ) -> ModelWrapperOutput:
        selected_foundation_names = self._normalize_name_subset(
            selected=foundation_names,
            available=self.foundation_modules,
            subset_name="foundation module",
        )
        selected_head_names = self._normalize_name_subset(
            selected=head_names,
            available=self.head_modules,
            subset_name="head module",
        )

        foundation_compute_grn = self._normalize_bool_selector(
            value=compute_grn,
            names=selected_foundation_names,
            option_name="compute_grn",
        )
        foundation_return_factors = self._normalize_bool_selector(
            value=return_factors,
            names=selected_foundation_names,
            option_name="return_factors",
        )

        for head_name in selected_head_names:
            foundation_name = self.head_to_foundation[head_name]
            if foundation_name not in selected_foundation_names:
                selected_foundation_names.append(foundation_name)
                foundation_compute_grn[foundation_name] = False
                foundation_return_factors[foundation_name] = True
            else:
                foundation_return_factors[foundation_name] = True

        foundations: dict[str, FoundationModuleOutput] = {}
        for foundation_name in selected_foundation_names:
            foundation_module = self.foundation_modules[foundation_name]
            output = foundation_module(
                tokens,
                return_factors=foundation_return_factors[foundation_name],
                compute_grn=foundation_compute_grn[foundation_name],
            )
            foundations[foundation_name] = self._coerce_foundation_output(
                output,
                return_factors=foundation_return_factors[foundation_name],
                compute_grn=foundation_compute_grn[foundation_name],
            )

        heads: dict[str, Any] = {}
        for head_name in selected_head_names:
            head = self.head_modules[head_name]
            foundation_name = self.head_to_foundation[head_name]
            foundation_output = foundations[foundation_name]
            head_kwargs = self._build_head_kwargs(
                head,
                tokens=tokens,
                foundation_output=foundation_output,
                foundation_name=foundation_name,
            )
            heads[head_name] = head(**head_kwargs)

        return ModelWrapperOutput(
            foundations=foundations,
            heads=heads,
        )
