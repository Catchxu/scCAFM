# src/sccafm/builder.py
from .models import SFM, ELBOLoss
from .loss import SFMLoss, DAGLoss, PriorLoss
from .tokenizer import TomeTokenizer, GeneTokenizer, ExprTokenizer, CondTokenizer, BatchTokenizer


# ----------------- Registries -----------------
MODEL_REGISTRY = {
    "sfm": SFM
}

TOKENIZER_REGISTRY = {
    "tome": TomeTokenizer,
    "gene": GeneTokenizer,
    "expr": ExprTokenizer,
    "cond": CondTokenizer,
    "batch": BatchTokenizer,
}

LOSS_REGISTRY = {
    "sfm": SFMLoss,
    "elbo": ELBOLoss,
    "dag": DAGLoss,
    "prior": PriorLoss
}


# ----------------- Build Functions -----------------
def build_model(cfg: dict, **runtime_kwargs):
    name = cfg["name"]
    kwargs = cfg.get("kwargs", {})

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'")
    
    kwargs.update(runtime_kwargs)
    return MODEL_REGISTRY[name](**kwargs)


def build_tokenizer(cfg: dict, **runtime_kwargs):
    name = cfg["name"]
    kwargs = cfg.get("kwargs", {})

    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer '{name}'")

    kwargs.update(runtime_kwargs)
    return TOKENIZER_REGISTRY[name](**kwargs)


def build_loss(cfg: dict, **runtime_kwargs):
    name = cfg["name"]
    kwargs = cfg.get("kwargs", {})

    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'")

    kwargs.update(runtime_kwargs)
    return LOSS_REGISTRY[name](**kwargs)
