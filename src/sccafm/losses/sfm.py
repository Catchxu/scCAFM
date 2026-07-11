from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from .dag import DAGLoss
from .elbo import ELBOLoss
from .prior import PriorLoss
from .reduction import distributed_any_nonfinite
from .sparsity import SparsityLoss
from ..models.wrapper import ModelWrapperOutput


@dataclass
class LossResult:
    total: torch.Tensor
    metrics: dict[str, float]
    disabled_regularizers: tuple[str, ...] = ()


class CosineValueSchedule:
    def __init__(self, initial: float, final: float, total_epochs: int) -> None:
        if total_epochs <= 0:
            raise ValueError(f"`total_epochs` must be positive, got {total_epochs}.")
        self.initial = float(initial)
        self.final = float(final)
        self.total_epochs = int(total_epochs)

    def value_at(self, epoch: int) -> float:
        progress = min(max(epoch, 0), self.total_epochs) / self.total_epochs
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final + (self.initial - self.final) * cosine


class PretrainingLossManager(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        token_dict: pd.DataFrame,
        total_epochs: int,
        total_steps: int,
    ) -> None:
        super().__init__()
        loss_cfg = config["loss"]
        self.foundation_name = loss_cfg.get("foundation_name", "sfm")
        self.head_name = loss_cfg.get("head_name", "vgae")

        self.use_elbo = bool(loss_cfg.get("elbo", {}).get("enabled", True))
        self.use_prior = bool(loss_cfg.get("prior", {}).get("enabled", False))
        self.use_dag = bool(loss_cfg.get("dag", {}).get("enabled", False))
        self.use_sparsity = bool(loss_cfg.get("sparsity", {}).get("enabled", False))
        policy = str(config.get("trainer", {}).get("nonfinite_policy", "disable_regularizer"))
        if policy != "disable_regularizer":
            raise ValueError(
                "`trainer.nonfinite_policy` must be 'disable_regularizer', "
                f"got {policy!r}."
            )

        self.elbo = ELBOLoss() if self.use_elbo else None
        scheduler_cfg = config.get("scheduler", {})
        warmup_ratio = float(scheduler_cfg.get("warmup_ratio", 0.03))
        warmup_steps = int(total_steps * warmup_ratio)

        dag_cfg = loss_cfg.get("dag", {})
        dag_kwargs = dict(dag_cfg.get("kwargs", {}))
        if self.use_dag and "warmup_steps" not in dag_kwargs:
            dag_kwargs["warmup_steps"] = warmup_steps
        self.dag = DAGLoss(**dag_kwargs) if self.use_dag else None

        sparsity_cfg = loss_cfg.get("sparsity", {})
        sparsity_kwargs = dict(sparsity_cfg.get("kwargs", {}))
        if self.use_sparsity and "warmup_steps" not in sparsity_kwargs:
            sparsity_kwargs["warmup_steps"] = warmup_steps
        self.sparsity = SparsityLoss(**sparsity_kwargs) if self.use_sparsity else None

        prior_cfg = loss_cfg.get("prior", {})
        prior_kwargs = dict(prior_cfg.get("kwargs", {}))
        prior_grn_path = prior_cfg.get("prior_grn_path")
        true_grn_df = None
        if prior_grn_path:
            true_grn_df = pd.read_csv(Path(prior_grn_path).expanduser().resolve())
        self.prior = (
            PriorLoss(
                token_dict=token_dict,
                true_grn_df=true_grn_df,
                **prior_kwargs,
            )
            if self.use_prior
            else None
        )

        schedule_cfg = prior_cfg.get("weight_schedule", {})
        self.prior_schedule = (
            CosineValueSchedule(
                initial=float(schedule_cfg.get("initial", 1.0)),
                final=float(schedule_cfg.get("final", 0.0)),
                total_epochs=total_epochs,
            )
            if self.use_prior
            else None
        )

    def forward(
        self,
        tokens: dict[str, torch.Tensor | None],
        model_output: ModelWrapperOutput,
        current_epoch: int,
        global_step: int = 0,
    ) -> LossResult:
        foundation_output = model_output.foundations[self.foundation_name]
        vgae_output = model_output.heads.get(self.head_name)

        total_loss: torch.Tensor | None = None
        metrics: dict[str, float] = {}
        core_values: list[torch.Tensor] = []
        disabled_regularizers: list[str] = []

        if self.use_elbo:
            if vgae_output is None:
                raise ValueError(f"Head output {self.head_name!r} is required for ELBO loss.")
            elbo_raw = self.elbo(tokens=tokens, vgae_output=vgae_output)
            self._require_fp32("elbo", elbo_raw)
            total_loss = elbo_raw if total_loss is None else total_loss + elbo_raw
            core_values.append(elbo_raw)
            metrics["elbo"] = float(elbo_raw.detach().item())

        if self.use_prior:
            prior_raw = self.prior(tokens=tokens, factors=foundation_output.factors)
            prior_weight = self.prior_schedule.value_at(current_epoch)
            weighted_prior = prior_raw * prior_weight
            self._require_fp32("prior", weighted_prior)
            total_loss = weighted_prior if total_loss is None else total_loss + weighted_prior
            core_values.append(weighted_prior)
            metrics["prior"] = float(weighted_prior.detach().item())

        if self.use_dag:
            dag_raw = self.dag(foundation_output.factors, global_step=global_step)
            self._require_fp32("dag", dag_raw)
            total_loss = dag_raw if total_loss is None else total_loss + dag_raw
            metrics["dag"] = float(self.dag.last_h.detach().item())
            metrics["dag_weighted"] = float(self.dag.last_weighted.detach().item())
            metrics["dag_active"] = float(self.dag.active.item())
            metrics["dag_disabled"] = float(self.dag.disabled.item())
            if bool(self.dag.just_disabled.item()):
                disabled_regularizers.append("dag")

        if self.use_sparsity:
            sparsity_raw = self.sparsity(
                tokens=tokens,
                factors=foundation_output.factors,
                global_step=global_step,
            )
            self._require_fp32("sparsity", sparsity_raw)
            total_loss = sparsity_raw if total_loss is None else total_loss + sparsity_raw
            metrics["sparsity"] = float(self.sparsity.last_mean.detach().item())
            metrics["sparsity_weighted"] = float(self.sparsity.last_weighted.detach().item())
            metrics["sparsity_active"] = float(self.sparsity.active.item())
            metrics["sparsity_disabled"] = float(self.sparsity.disabled.item())
            if bool(self.sparsity.just_disabled.item()):
                disabled_regularizers.append("sparsity")

        if total_loss is None:
            raise ValueError("At least one loss component must be enabled.")
        self._require_fp32("total", total_loss)
        if distributed_any_nonfinite(*core_values, total_loss):
            raise FloatingPointError(
                f"Non-finite core SFM loss detected at global_step={int(global_step)}."
            )
        return LossResult(
            total=total_loss,
            metrics=metrics,
            disabled_regularizers=tuple(disabled_regularizers),
        )

    @staticmethod
    def _require_fp32(name: str, value: torch.Tensor) -> None:
        if value.dtype != torch.float32:
            raise TypeError(f"SFM {name} loss must be fp32, got {value.dtype}.")

    @torch.no_grad()
    def disable_next_regularizer(self) -> str | None:
        if self.dag is not None and not bool(self.dag.disabled.item()):
            self.dag.disabled.fill_(True)
            self.dag.active.zero_()
            return "dag"
        if self.sparsity is not None and not bool(self.sparsity.disabled.item()):
            self.sparsity.disabled.fill_(True)
            self.sparsity.active.zero_()
            return "sparsity"
        return None
