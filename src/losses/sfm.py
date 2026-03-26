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
from ..models.wrapper import ModelWrapperOutput


@dataclass
class LossResult:
    total: torch.Tensor
    metrics: dict[str, float]


class CosineValueSchedule:
    def __init__(self, initial: float, final: float, span_epochs: int) -> None:
        if span_epochs <= 0:
            raise ValueError(f"`span_epochs` must be positive, got {span_epochs}.")
        self.initial = float(initial)
        self.final = float(final)
        self.span_epochs = int(span_epochs)

    def value_at(self, epoch: int) -> float:
        progress = min(max(epoch, 0), self.span_epochs) / self.span_epochs
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final + (self.initial - self.final) * cosine


class PretrainingLossManager(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        token_dict: pd.DataFrame,
        total_epochs: int,
    ) -> None:
        super().__init__()
        loss_cfg = config["loss"]
        self.foundation_name = loss_cfg.get("foundation_name", "sfm")
        self.head_name = loss_cfg.get("head_name", "vgae")

        self.use_elbo = bool(loss_cfg.get("elbo", {}).get("enabled", True))
        self.use_prior = bool(loss_cfg.get("prior", {}).get("enabled", False))
        self.use_dag = bool(loss_cfg.get("dag", {}).get("enabled", False))

        self.elbo = ELBOLoss() if self.use_elbo else None
        self.dag = DAGLoss(**loss_cfg.get("dag", {}).get("kwargs", {})) if self.use_dag else None

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
        span_epochs = int(schedule_cfg.get("span_epochs") or total_epochs)
        self.prior_schedule = (
            CosineValueSchedule(
                initial=float(schedule_cfg.get("initial", 1.0)),
                final=float(schedule_cfg.get("final", 0.0)),
                span_epochs=span_epochs,
            )
            if self.use_prior
            else None
        )

    def forward(
        self,
        tokens: dict[str, torch.Tensor | None],
        model_output: ModelWrapperOutput,
        current_epoch: int,
    ) -> LossResult:
        foundation_output = model_output.foundations[self.foundation_name]
        vgae_output = model_output.heads.get(self.head_name)

        total_loss: torch.Tensor | None = None
        metrics: dict[str, float] = {}

        if self.use_elbo:
            if vgae_output is None:
                raise ValueError(f"Head output {self.head_name!r} is required for ELBO loss.")
            elbo_raw = self.elbo(tokens=tokens, vgae_output=vgae_output)
            total_loss = elbo_raw if total_loss is None else total_loss + elbo_raw
            metrics["elbo"] = float(elbo_raw.detach().item())

        if self.use_prior:
            prior_raw = self.prior(tokens=tokens, factors=foundation_output.factors)
            prior_weight = self.prior_schedule.value_at(current_epoch)
            weighted_prior = prior_raw * prior_weight
            total_loss = weighted_prior if total_loss is None else total_loss + weighted_prior
            metrics["prior"] = float(weighted_prior.detach().item())

        if self.use_dag:
            dag_raw = self.dag(foundation_output.factors)
            total_loss = dag_raw if total_loss is None else total_loss + dag_raw
            metrics["dag"] = float(dag_raw.detach().item())

        if total_loss is None:
            raise ValueError("At least one loss component must be enabled.")
        return LossResult(total=total_loss, metrics=metrics)
