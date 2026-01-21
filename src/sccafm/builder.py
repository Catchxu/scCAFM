# src/sccafm/builder.py
from .models import SFM, ELBOLoss
from .loss import SFMLoss, DAGLoss, PriorLoss
from .tokenizer import TomeTokenizer, GeneTokenizer, ExprTokenizer, CondTokenizer, BatchTokenizer


# ----------------- Registries -----------------
MODEL_REGISTRY = {
    "sfm": SFM
}

LOSS_REGISTRY = {
    "sfm": SFMLoss,
    "elbo": ELBOLoss,
    "dag": DAGLoss,
    "prior": PriorLoss
}

TOKENIZER_REGISTRY = {
    "tome": TomeTokenizer,
    "gene": GeneTokenizer,
    "expr": ExprTokenizer,
    "cond": CondTokenizer,
    "batch": BatchTokenizer,
}


# ----------------- Build Functions -----------------
def build_model(cfg: dict):
    name = cfg["name"]
    kwargs = cfg.get("kwargs", {})

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'")

    return MODEL_REGISTRY[name](**kwargs)

def build_loss(cfg: dict):
    name = cfg["name"]
    kwargs = cfg.get("kwargs", {})

    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'")

    return LOSS_REGISTRY[name](**kwargs)

def build_tokenizer(cfg: dict):
    name = cfg["name"]
    kwargs = cfg.get("kwargs", {})

    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer '{name}'")

    return TOKENIZER_REGISTRY[name](**kwargs)